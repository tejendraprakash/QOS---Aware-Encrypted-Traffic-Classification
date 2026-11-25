"""
dashboard/pipeline_runner.py

Robust, thread-safe execution interface for the QoS-Aware Encrypted Traffic Pipeline.
Integrates the end-to-end research verification engine with the Streamlit dashboard:
- PCAP discovery and validation in data/raw
- Secure upload and sanitization of user-provided PCAPs
- Thread-safe and inter-process concurrency locking
- Real-time streaming and buffered capture of execution telemetry
- Comprehensive error handling and output artifact verification
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Base project directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "runtime_verification"
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_pipeline_end_to_end.py"

# Valid magic byte signatures for PCAP and PCAPNG files
PCAP_MAGIC_SIGNATURES = {
    b"\xd4\xc3\xb2\xa1": "PCAP (Microseconds, Little-Endian)",
    b"\xa1\xb2\xc3\xd4": "PCAP (Microseconds, Big-Endian)",
    b"\x4d\x3c\xb2\xa1": "PCAP (Nanoseconds, Little-Endian)",
    b"\xa1\xb2\x3c\x4d": "PCAP (Nanoseconds, Big-Endian)",
    b"\x0a\x0d\x0d\x0a": "PCAPNG (Section Header Block)",
}

# Concurrency Locks
_THREAD_LOCK = threading.Lock()
LOCK_FILE_PATH = PROJECT_ROOT / ".pipeline_execution.lock"
MAX_LOCK_STALE_AGE_SEC = 900.0  # 15 minutes max execution timeout before treating lock as stale


class PipelineBusyError(Exception):
    """Raised when an execution is attempted while another pipeline run is in progress."""
    pass


class PipelineExecutionError(Exception):
    """Raised when the pipeline sub-process fails or exits with a non-zero return code."""
    pass


class PipelineLock:
    """
    Dual concurrency lock: Python threading.Lock + Atomic Cross-Process File Lock.
    Prevents race conditions between concurrent Streamlit sessions or multi-worker processes.
    """

    def __init__(self, lock_path: Path = LOCK_FILE_PATH, timeout_sec: float = 0.0):
        self.lock_path = lock_path
        self.timeout_sec = timeout_sec
        self._fd: Optional[int] = None
        self._thread_acquired = False

    def acquire(self, blocking: bool = False) -> bool:
        start_time = time.time()
        while True:
            # 1. Acquire in-process thread lock
            if not self._thread_acquired:
                self._thread_acquired = _THREAD_LOCK.acquire(blocking=False)
                if not self._thread_acquired:
                    if not blocking or (time.time() - start_time) >= self.timeout_sec:
                        return False
                    time.sleep(0.1)
                    continue

            # 2. Check for stale lock file
            if self.lock_path.exists():
                try:
                    mtime = self.lock_path.stat().st_mtime
                    if (time.time() - mtime) > MAX_LOCK_STALE_AGE_SEC:
                        logger.warning(f"Removing stale pipeline lock file ({time.time() - mtime:.1f}s old)")
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                except OSError:
                    pass

            # 3. Acquire atomic file lock
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                # Write current process ID and timestamp
                os.write(self._fd, f"pid={os.getpid()};time={time.time()}".encode("utf-8"))
                return True
            except FileExistsError:
                if not blocking or (time.time() - start_time) >= self.timeout_sec:
                    if self._thread_acquired:
                        _THREAD_LOCK.release()
                        self._thread_acquired = False
                    return False
                time.sleep(0.1)
            except OSError as e:
                logger.error(f"Error acquiring pipeline lock file: {e}")
                if self._thread_acquired:
                    _THREAD_LOCK.release()
                    self._thread_acquired = False
                return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        if self.lock_path.exists():
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass

        if self._thread_acquired:
            try:
                _THREAD_LOCK.release()
            except RuntimeError:
                pass
            self._thread_acquired = False

    def __enter__(self):
        if not self.acquire(blocking=self.timeout_sec > 0):
            raise PipelineBusyError("Another pipeline execution is already active.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def is_pipeline_running() -> bool:
    """Check if a pipeline run is currently active (either in-thread or via file lock)."""
    if _THREAD_LOCK.locked():
        return True
    if LOCK_FILE_PATH.exists():
        try:
            mtime = LOCK_FILE_PATH.stat().st_mtime
            if (time.time() - mtime) < MAX_LOCK_STALE_AGE_SEC:
                return True
        except OSError:
            pass
    return False


# -----------------------------------------------------------------------------
# 1. PCAP Discovery & Validation
# -----------------------------------------------------------------------------

def validate_pcap_file(pcap_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate that a given path points to an accessible, non-empty PCAP or PCAPNG file
    with recognized binary magic headers.

    Returns:
        (is_valid: bool, message: str)
    """
    path = Path(pcap_path)
    if not path.exists():
        return False, f"File does not exist: {path}"
    if not path.is_file():
        return False, f"Path is not a regular file: {path}"

    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"Unable to read file metadata: {e}"

    if size == 0:
        return False, f"File is empty (0 bytes): {path.name}"

    suffix = path.suffix.lower()
    if suffix not in [".pcap", ".pcapng"]:
        return False, f"Unsupported file extension '{suffix}'. Expected .pcap or .pcapng."

    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if len(header) < 4:
            return False, f"File is too short to contain a valid PCAP header ({len(header)} bytes)."

        if header not in PCAP_MAGIC_SIGNATURES:
            return False, (
                f"Invalid file magic bytes: 0x{header.hex()}. "
                f"Does not match standard PCAP (0xd4c3b2a1 / 0xa1b2c3d4) or PCAPNG (0x0a0d0d0a)."
            )
    except Exception as e:
        return False, f"Failed reading file header: {e}"

    return True, f"Valid {PCAP_MAGIC_SIGNATURES[header]} file ({size / (1024 * 1024):.2f} MB)"


def get_available_pcaps(
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    as_strings: bool = False,
    validate_magic: bool = True,
) -> Union[List[Path], List[str]]:
    """
    Discover all valid PCAP and PCAPNG capture files in the specified directory.

    Args:
        raw_dir: Directory containing capture files (defaults to data/raw).
        as_strings: If True, returns list of string paths; if False, returns list of Path objects.
        validate_magic: If True, filters files by binary magic header checks.

    Returns:
        List of Path objects (or strings) sorted alphabetically.
    """
    directory = Path(raw_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory

    if not directory.exists() or not directory.is_dir():
        logger.warning(f"PCAP directory not found: {directory}")
        return []

    valid_files: List[Path] = []
    candidates = list(directory.glob("*.pcap")) + list(directory.glob("*.pcapng"))

    for file_path in candidates:
        if not file_path.is_file():
            continue
        if validate_magic:
            is_valid, _ = validate_pcap_file(file_path)
            if is_valid:
                valid_files.append(file_path)
        else:
            if file_path.stat().st_size > 0:
                valid_files.append(file_path)

    # Sort files alphabetically by name
    valid_files.sort(key=lambda p: p.name.lower())

    if as_strings:
        return [str(p) for p in valid_files]
    return valid_files


def get_pcap_metadata(pcap_path: Union[str, Path]) -> Dict[str, Any]:
    """Retrieve detailed metadata for a PCAP file."""
    path = Path(pcap_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return {
            "name": path.name,
            "filename": path.name,
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "size_mb": 0.0,
            "is_valid": False,
            "magic_hex": "N/A",
            "message": "File does not exist",
        }

    is_valid, message = validate_pcap_file(path)
    size_bytes = path.stat().st_size if path.exists() else 0
    mtime = path.stat().st_mtime if path.exists() else 0

    magic_hex = "N/A"
    try:
        with open(path, "rb") as f:
            hdr = f.read(4)
            if len(hdr) >= 4:
                magic_hex = f"0x{hdr.hex()}"
    except Exception:
        pass

    return {
        "name": path.name,
        "filename": path.name,
        "path": str(path),
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "extension": path.suffix.lower(),
        "is_valid": is_valid,
        "magic_hex": magic_hex,
        "message": message,
        "modified_timestamp": mtime,
    }


# -----------------------------------------------------------------------------
# 2. Upload Handler
# -----------------------------------------------------------------------------

def save_uploaded_pcap(
    uploaded_file: Any,
    target_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    prefix: str = "uploaded_",
    validate: bool = True,
) -> Path:
    """
    Safely persist an uploaded file from Streamlit's st.file_uploader into data/raw.

    Args:
        uploaded_file: Streamlit UploadedFile, bytes, or file-like object with .name and .getbuffer()/.read().
        target_dir: Destination directory (defaults to data/raw).
        prefix: Filename prefix to differentiate uploaded traces (defaults to 'uploaded_').
        validate: Whether to perform PCAP magic byte verification on the saved file.

    Returns:
        Path to the saved PCAP file.

    Raises:
        ValueError: If filename is invalid, file is empty, or magic bytes check fails.
        IOError: If saving to disk encounters permission or write errors.
    """
    dest_dir = Path(target_dir)
    if not dest_dir.is_absolute():
        dest_dir = PROJECT_ROOT / dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract raw filename safely
    raw_name = getattr(uploaded_file, "name", "custom_capture.pcap")
    base_name = Path(raw_name).name  # Prevent directory traversal attacks

    # 2. Sanitize filename: allow alphanumeric, underscore, hyphen, and dot
    clean_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(base_name).stem)
    clean_suffix = Path(base_name).suffix.lower()
    if clean_suffix not in [".pcap", ".pcapng"]:
        clean_suffix = ".pcap"

    # Ensure prefix
    if not clean_stem.startswith(prefix):
        final_filename = f"{prefix}{clean_stem}{clean_suffix}"
    else:
        final_filename = f"{clean_stem}{clean_suffix}"

    target_path = dest_dir / final_filename

    # 3. Write data to destination
    try:
        if hasattr(uploaded_file, "getbuffer"):
            # Streamlit UploadedFile
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif hasattr(uploaded_file, "read"):
            # File-like object
            with open(target_path, "wb") as f:
                content = uploaded_file.read()
                f.write(content if isinstance(content, bytes) else content.encode("utf-8"))
        elif isinstance(uploaded_file, (bytes, bytearray)):
            with open(target_path, "wb") as f:
                f.write(uploaded_file)
        else:
            raise ValueError(f"Unsupported uploaded_file type: {type(uploaded_file)}")
    except Exception as e:
        logger.error(f"Failed to write uploaded file to {target_path}: {e}")
        raise IOError(f"Could not save uploaded PCAP to {target_path}: {e}") from e

    # 4. Post-write validation
    if validate:
        is_valid, reason = validate_pcap_file(target_path)
        if not is_valid:
            # Clean up corrupted or invalid upload
            target_path.unlink(missing_ok=True)
            raise ValueError(f"Uploaded file failed PCAP validation: {reason}")

    logger.info(f"Successfully saved uploaded PCAP to {target_path} ({target_path.stat().st_size} bytes)")
    return target_path


# -----------------------------------------------------------------------------
# 3. Pipeline Execution Engine
# -----------------------------------------------------------------------------

def parse_simulation_metrics(metrics_csv_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse generated e2e_simulation_metrics.csv into a list of structured metric records."""
    csv_path = Path(metrics_csv_path)
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error parsing metrics CSV {csv_path}: {e}")
        return []


def execute_pipeline(
    pcap_path: Union[str, Path],
    regime: str = "Moderate",
    sim_time: float = 30.0,
    model_path: Optional[Union[str, Path]] = None,
    outdir: Optional[Union[str, Path]] = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    timeout_sec: float = 300.0,
    log_callback: Optional[Callable[[str], None]] = None,
    blocking_lock: bool = False,
) -> Dict[str, Any]:
    """
    Safely execute the end-to-end QoS verification pipeline in a child process.

    Args:
        pcap_path: Path to target .pcap or .pcapng file.
        regime: Congestion regime ('Low', 'Moderate', 'High').
        sim_time: Simulation verification duration in seconds.
        model_path: Optional path to custom trained XGBoost model JSON.
        outdir: Directory to save generated verification artifacts.
        seed: Random seed for network simulation.
        timeout_sec: Max allowed execution duration before timeout termination.
        log_callback: Optional real-time callback receiving stdout/stderr line strings.
        blocking_lock: If True, waits for active runs to finish; if False, returns failure immediately if busy.

    Returns:
        Structured result dictionary:
        {
            "status": "success" | "failure",
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "execution_duration": float,
            "artifacts": {
                "inferred_flows": Path,
                "e2e_metrics": Path,
                "extracted_flows": Path,
            },
            "summary": { ... },
            "error": Optional[str],
        }
    """
    t_start = time.time()

    # 1. Validate inputs
    pcap = Path(pcap_path)
    if not pcap.is_absolute():
        pcap = PROJECT_ROOT / pcap

    if not pcap.exists():
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Target PCAP file does not exist: {pcap}",
            "execution_duration": 0.0,
            "artifacts": {},
            "summary": {},
            "error": f"PCAP not found: {pcap}",
        }

    valid_regimes = ["Low", "Moderate", "High"]
    if regime not in valid_regimes:
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Invalid regime '{regime}'. Expected one of {valid_regimes}.",
            "execution_duration": 0.0,
            "artifacts": {},
            "summary": {},
            "error": f"Invalid regime: {regime}",
        }

    if sim_time <= 0:
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Invalid sim_time {sim_time}. Must be positive.",
            "execution_duration": 0.0,
            "artifacts": {},
            "summary": {},
            "error": "sim_time must be positive",
        }

    out_directory = Path(outdir or DEFAULT_OUTPUT_DIR)
    if not out_directory.is_absolute():
        out_directory = PROJECT_ROOT / out_directory
    out_directory.mkdir(parents=True, exist_ok=True)

    script_path = DEFAULT_SCRIPT_PATH
    if not script_path.exists():
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "",
            "stderr": f"Pipeline script not found at {script_path}",
            "execution_duration": 0.0,
            "artifacts": {},
            "summary": {},
            "error": f"Script not found: {script_path}",
        }

    # 2. Acquire concurrency lock
    lock = PipelineLock()
    if not lock.acquire(blocking=blocking_lock):
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "",
            "stderr": "Pipeline concurrency lock active: another simulation run is currently in progress.",
            "execution_duration": 0.0,
            "artifacts": {},
            "summary": {},
            "error": "Pipeline busy: concurrent execution blocked.",
        }

    # 3. Construct command arguments safely without shell expansion
    cmd = [
        sys.executable,
        "-u",  # Unbuffered binary stdout and stderr for live telemetry
        str(script_path),
        "--pcap", str(pcap),
        "--regime", regime,
        "--sim-time", str(sim_time),
        "--outdir", str(out_directory),
        "--seed", str(seed),
    ]

    if model_path:
        model_p = Path(model_path)
        if not model_p.is_absolute():
            model_p = PROJECT_ROOT / model_p
        if model_p.exists():
            cmd.extend(["--model", str(model_p)])
        else:
            logger.warning(f"Specified model path {model_p} not found, falling back to default.")

    # Track output logs
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    import subprocess

    proc: Optional[subprocess.Popen] = None
    timed_out = False

    try:
        # Launch sub-process
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified chronological streaming
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Stream output in real time
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            stdout_lines.append(line)
            if log_callback:
                try:
                    log_callback(line)
                except Exception as cb_err:
                    logger.debug(f"Log callback error: {cb_err}")

            # Check timeout
            if (time.time() - t_start) > timeout_sec:
                timed_out = True
                proc.terminate()
                break

        # Wait for termination
        if timed_out:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.wait(timeout=max(5.0, timeout_sec - (time.time() - t_start)))

    except subprocess.TimeoutExpired:
        timed_out = True
        if proc:
            proc.kill()
            proc.wait()
    except Exception as e:
        logger.error(f"Unexpected error executing pipeline process: {e}")
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": "".join(stdout_lines),
            "stderr": str(e),
            "execution_duration": time.time() - t_start,
            "artifacts": {},
            "summary": {},
            "error": str(e),
        }
    finally:
        # Close stdout pipe if open
        if proc and proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        # Always release the concurrency lock
        lock.release()

    duration = time.time() - t_start
    full_stdout = "".join(stdout_lines)
    full_stderr = "".join(stderr_lines)

    if timed_out:
        return {
            "status": "failure",
            "returncode": -1,
            "stdout": full_stdout,
            "stderr": f"Pipeline execution timed out after {timeout_sec:.1f} seconds.",
            "execution_duration": duration,
            "artifacts": {},
            "summary": {},
            "error": f"Timeout exceeded ({timeout_sec}s)",
        }

    returncode = proc.returncode if proc else -1

    # 4. Check and verify output artifacts
    inferred_flows_path = out_directory / "inferred_flows.csv"
    e2e_metrics_path = out_directory / "e2e_simulation_metrics.csv"
    extracted_flows_path = out_directory / "extracted_flows" / "flow_features.csv"

    artifacts: Dict[str, Path] = {}
    if inferred_flows_path.exists():
        artifacts["inferred_flows"] = inferred_flows_path
    if e2e_metrics_path.exists():
        artifacts["e2e_metrics"] = e2e_metrics_path
    if extracted_flows_path.exists():
        artifacts["extracted_flows"] = extracted_flows_path

    # Check for success criteria
    is_success = (returncode == 0) and (inferred_flows_path.exists()) and (e2e_metrics_path.exists())

    # 5. Extract structured summary if successful
    summary: Dict[str, Any] = {}
    if is_success:
        try:
            metrics_list = parse_simulation_metrics(e2e_metrics_path)
            summary["metrics"] = metrics_list
            if inferred_flows_path.exists():
                df_inf = pd.read_csv(inferred_flows_path)
                summary["num_flows"] = len(df_inf)
                summary["class_counts"] = df_inf["pred_class"].value_counts().to_dict() if "pred_class" in df_inf.columns else {}
                if "confidence" in df_inf.columns:
                    summary["mean_confidence"] = float(df_inf["confidence"].mean())
        except Exception as e:
            logger.warning(f"Error compiling summary metrics: {e}")

    return {
        "status": "success" if is_success else "failure",
        "returncode": returncode,
        "stdout": full_stdout,
        "stderr": full_stderr if full_stderr else ("" if is_success else "Process terminated with error"),
        "execution_duration": round(duration, 3),
        "artifacts": artifacts,
        "summary": summary,
        "error": None if is_success else f"Pipeline exited with return code {returncode}",
    }


def execute_pipeline_stream(
    pcap_path: Union[str, Path],
    regime: str = "Moderate",
    sim_time: float = 30.0,
    model_path: Optional[Union[str, Path]] = None,
    outdir: Optional[Union[str, Path]] = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    timeout_sec: float = 300.0,
    blocking_lock: bool = False,
) -> Generator[Tuple[str, Any], None, None]:
    """
    Generator streaming pipeline execution events in real time.

    Yields tuples of (event_type, data):
        ("log", line_str) -> Emitted on each stdout line from the execution.
        ("complete", result_dict) -> Emitted once at completion with full status and artifacts.

    Example:
        for event, data in execute_pipeline_stream("data/raw/scpDown4.pcap"):
            if event == "log":
                st.text(data)
            elif event == "complete":
                st.success(f"Execution finished with status {data['status']}")
    """
    buffer_lines: List[str] = []

    def on_line(line: str):
        buffer_lines.append(line)

    # We run the sub-process generator directly to yield lines as they happen
    t_start = time.time()
    pcap = Path(pcap_path)
    if not pcap.is_absolute():
        pcap = PROJECT_ROOT / pcap

    if not pcap.exists():
        yield ("log", f"[ERROR] PCAP file not found: {pcap}\n")
        yield (
            "complete",
            {
                "status": "failure",
                "returncode": -1,
                "stdout": "",
                "stderr": f"Target PCAP file does not exist: {pcap}",
                "execution_duration": 0.0,
                "artifacts": {},
                "summary": {},
                "error": f"PCAP not found: {pcap}",
            },
        )
        return

    lock = PipelineLock()
    if not lock.acquire(blocking=blocking_lock):
        yield ("log", "[ERROR] Another pipeline execution is currently active. Concurrency lock denied.\n")
        yield (
            "complete",
            {
                "status": "failure",
                "returncode": -1,
                "stdout": "",
                "stderr": "Pipeline concurrency lock active: another run is currently executing.",
                "execution_duration": 0.0,
                "artifacts": {},
                "summary": {},
                "error": "Pipeline busy: concurrent execution blocked.",
            },
        )
        return

    out_directory = Path(outdir or DEFAULT_OUTPUT_DIR)
    if not out_directory.is_absolute():
        out_directory = PROJECT_ROOT / out_directory
    out_directory.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(DEFAULT_SCRIPT_PATH),
        "--pcap", str(pcap),
        "--regime", regime,
        "--sim-time", str(sim_time),
        "--outdir", str(out_directory),
        "--seed", str(seed),
    ]
    if model_path:
        model_p = Path(model_path)
        if not model_p.is_absolute():
            model_p = PROJECT_ROOT / model_p
        if model_p.exists():
            cmd.extend(["--model", str(model_p)])

    import subprocess
    proc = None
    timed_out = False

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            buffer_lines.append(line)
            yield ("log", line)

            if (time.time() - t_start) > timeout_sec:
                timed_out = True
                proc.terminate()
                break

        if timed_out:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.wait(timeout=max(5.0, timeout_sec - (time.time() - t_start)))

    except Exception as e:
        yield ("log", f"[ERROR] Pipeline execution exception: {e}\n")
        yield (
            "complete",
            {
                "status": "failure",
                "returncode": -1,
                "stdout": "".join(buffer_lines),
                "stderr": str(e),
                "execution_duration": time.time() - t_start,
                "artifacts": {},
                "summary": {},
                "error": str(e),
            },
        )
        return
    finally:
        if proc and proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        lock.release()

    duration = time.time() - t_start
    returncode = proc.returncode if proc else -1

    inferred_flows_path = out_directory / "inferred_flows.csv"
    e2e_metrics_path = out_directory / "e2e_simulation_metrics.csv"
    extracted_flows_path = out_directory / "extracted_flows" / "flow_features.csv"

    artifacts: Dict[str, Path] = {}
    if inferred_flows_path.exists():
        artifacts["inferred_flows"] = inferred_flows_path
    if e2e_metrics_path.exists():
        artifacts["e2e_metrics"] = e2e_metrics_path
    if extracted_flows_path.exists():
        artifacts["extracted_flows"] = extracted_flows_path

    is_success = (returncode == 0) and inferred_flows_path.exists() and e2e_metrics_path.exists()

    summary: Dict[str, Any] = {}
    if is_success:
        try:
            summary["metrics"] = parse_simulation_metrics(e2e_metrics_path)
            if inferred_flows_path.exists():
                df_inf = pd.read_csv(inferred_flows_path)
                summary["num_flows"] = len(df_inf)
                summary["class_counts"] = df_inf["pred_class"].value_counts().to_dict() if "pred_class" in df_inf.columns else {}
                if "confidence" in df_inf.columns:
                    summary["mean_confidence"] = float(df_inf["confidence"].mean())
        except Exception as e:
            logger.warning(f"Error compiling summary metrics: {e}")

    final_result = {
        "status": "success" if is_success else "failure",
        "returncode": returncode,
        "stdout": "".join(buffer_lines),
        "stderr": "" if is_success else f"Exited with code {returncode}",
        "execution_duration": round(duration, 3),
        "artifacts": artifacts,
        "summary": summary,
        "error": None if is_success else f"Pipeline exited with return code {returncode}",
    }

    yield ("complete", final_result)
