"""
tests/test_pipeline_runner.py

Unit and integration tests for dashboard/pipeline_runner.py.
Verifies PCAP discovery, secure upload handling, concurrency locking,
real-time log streaming, and end-to-end execution.
"""

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.pipeline_runner import (
    PipelineLock,
    PipelineBusyError,
    get_available_pcaps,
    get_pcap_metadata,
    is_pipeline_running,
    save_uploaded_pcap,
    validate_pcap_file,
    execute_pipeline,
    execute_pipeline_stream,
    parse_simulation_metrics,
)

class MockUploadedFile:
    """Mock for Streamlit UploadedFile object."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return memoryview(self._data)

    def read(self):
        return self._data


class TestPipelineRunner(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Standard PCAP header: 0xd4c3b2a1 followed by 20 bytes standard header
        self.valid_pcap_bytes = bytes.fromhex("d4c3b2a10200040000000000000000000000040001000000")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_available_pcaps(self):
        # 1. Test against real data/raw
        pcaps = get_available_pcaps(PROJECT_ROOT / "data" / "raw")
        self.assertGreater(len(pcaps), 0)
        self.assertTrue(all(isinstance(p, Path) for p in pcaps))
        
        # Test as_strings=True
        pcap_strings = get_available_pcaps(PROJECT_ROOT / "data" / "raw", as_strings=True)
        self.assertGreater(len(pcap_strings), 0)
        self.assertTrue(all(isinstance(s, str) for s in pcap_strings))

        # Check scpDown4.pcap is discovered
        names = [p.name for p in pcaps]
        self.assertIn("scpDown4.pcap", names)

    def test_validate_pcap_file(self):
        # Valid PCAP file
        valid_file = self.raw_dir / "test.pcap"
        with open(valid_file, "wb") as f:
            f.write(self.valid_pcap_bytes)
        is_val, msg = validate_pcap_file(valid_file)
        self.assertTrue(is_val)
        self.assertIn("PCAP", msg)

        # Invalid magic bytes
        bad_magic_file = self.raw_dir / "bad.pcap"
        with open(bad_magic_file, "wb") as f:
            f.write(b"NOT_A_PCAP_HEADER_DATA")
        is_val, msg = validate_pcap_file(bad_magic_file)
        self.assertFalse(is_val)
        self.assertIn("Invalid file magic bytes", msg)

        # Empty file
        empty_file = self.raw_dir / "empty.pcap"
        empty_file.touch()
        is_val, msg = validate_pcap_file(empty_file)
        self.assertFalse(is_val)
        self.assertIn("empty", msg.lower())

        # Non-existent file
        is_val, msg = validate_pcap_file(self.raw_dir / "non_existent.pcap")
        self.assertFalse(is_val)

    def test_get_pcap_metadata(self):
        valid_file = self.raw_dir / "test.pcap"
        with open(valid_file, "wb") as f:
            f.write(self.valid_pcap_bytes)

        meta = get_pcap_metadata(valid_file)
        self.assertEqual(meta["name"], "test.pcap")
        self.assertTrue(meta["exists"])
        self.assertTrue(meta["is_valid"])
        self.assertEqual(meta["extension"], ".pcap")

    def test_save_uploaded_pcap(self):
        mock_file = MockUploadedFile("my_trace.pcap", self.valid_pcap_bytes)
        saved_path = save_uploaded_pcap(mock_file, target_dir=self.raw_dir)
        self.assertTrue(saved_path.exists())
        self.assertEqual(saved_path.name, "uploaded_my_trace.pcap")

        # Test directory traversal attack prevention
        attack_file = MockUploadedFile("../../../etc/shadow.pcap", self.valid_pcap_bytes)
        saved_attack = save_uploaded_pcap(attack_file, target_dir=self.raw_dir)
        self.assertTrue(saved_attack.exists())
        self.assertEqual(saved_attack.parent.resolve(), self.raw_dir.resolve())
        self.assertEqual(saved_attack.name, "uploaded_shadow.pcap")

        # Test invalid content rejection
        bad_file = MockUploadedFile("corrupt.pcap", b"corrupted bytes")
        with self.assertRaises(ValueError):
            save_uploaded_pcap(bad_file, target_dir=self.raw_dir, validate=True)

    def test_concurrency_lock(self):
        lock_file = Path(self.temp_dir) / "test.lock"
        lock1 = PipelineLock(lock_path=lock_file)
        lock2 = PipelineLock(lock_path=lock_file)

        # Lock 1 acquires
        self.assertTrue(lock1.acquire(blocking=False))

        # Lock 2 must fail to acquire non-blocking
        self.assertFalse(lock2.acquire(blocking=False))

        # Lock 1 releases
        lock1.release()

        # Lock 2 can now acquire
        self.assertTrue(lock2.acquire(blocking=False))
        lock2.release()

    def test_execute_pipeline_live(self):
        # Test real pipeline run on scpDown4.pcap with small sim_time
        test_outdir = Path(self.temp_dir) / "verification_output"
        logs = []

        def log_cb(line):
            logs.append(line)

        result = execute_pipeline(
            pcap_path=PROJECT_ROOT / "data" / "raw" / "scpDown4.pcap",
            regime="Moderate",
            sim_time=2.0,
            outdir=test_outdir,
            seed=42,
            timeout_sec=60.0,
            log_callback=log_cb,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["returncode"], 0)
        self.assertGreater(len(logs), 5)
        self.assertIn("inferred_flows", result["artifacts"])
        self.assertIn("e2e_metrics", result["artifacts"])
        self.assertTrue(result["artifacts"]["inferred_flows"].exists())
        self.assertTrue(result["artifacts"]["e2e_metrics"].exists())
        self.assertIn("metrics", result["summary"])
        self.assertGreater(result["summary"]["num_flows"], 0)

    def test_execute_pipeline_streaming(self):
        test_outdir = Path(self.temp_dir) / "stream_output"
        stream_logs = []
        final_result = None

        for event, data in execute_pipeline_stream(
            pcap_path=PROJECT_ROOT / "data" / "raw" / "scpDown4.pcap",
            regime="Moderate",
            sim_time=2.0,
            outdir=test_outdir,
            seed=42,
        ):
            if event == "log":
                stream_logs.append(data)
            elif event == "complete":
                final_result = data

        self.assertIsNotNone(final_result)
        self.assertEqual(final_result["status"], "success")
        self.assertGreater(len(stream_logs), 5)

    def test_execute_pipeline_invalid_inputs(self):
        # Non-existent PCAP
        res = execute_pipeline("data/raw/non_existent_file_xyz.pcap")
        self.assertEqual(res["status"], "failure")
        self.assertIn("does not exist", res["stderr"])

        # Invalid regime
        res = execute_pipeline(PROJECT_ROOT / "data" / "raw" / "scpDown4.pcap", regime="ExtremeCongestion")
        self.assertEqual(res["status"], "failure")
        self.assertIn("Invalid regime", res["stderr"])

        # Invalid sim_time
        res = execute_pipeline(PROJECT_ROOT / "data" / "raw" / "scpDown4.pcap", sim_time=-10.0)
        self.assertEqual(res["status"], "failure")
        self.assertIn("sim_time", res["stderr"])


if __name__ == "__main__":
    unittest.main()
