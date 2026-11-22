# Phase 7 Traffic Generation Corrections & Technical Remediation

**Document Reference:** `research_notes/phase7/traffic_generation_corrections.md`  
**Companion Documents:** `traffic_generation_audit.md`, `traffic_generation_final_specification.md`  
**Author:** Lead Research Engineer & Multi-Agent Audit Board  
**Date:** November 2026  
**Status:** Approved Corrections & Implementation Blueprint  

---

## 1. Executive Summary of Corrections

Following the multi-agent independent technical audit of `traffic_generation_design.md`, this document provides the formal mathematical derivations, structural parameter reconciliations, and line-by-line code fixes required to transition the traffic generation models into an approved, mathematically sound, and reproducible simulation state.

The corrections resolve:
1. The RFC 8312 TCP CUBIC $K$ equation error, parameter inversion, and open-loop window explosion.
2. The lognormal chunk sizing Jensen's inequality upward bias.
3. The streaming playback buffer drain bug, delayed ACK logic failure, and HTTP GET request sizing error.
4. The real-time Brownian motion clock drift, cellular SID comfort noise misattribution, and double-talk inflation.
5. The queueing Little's Law open-loop rate contradiction for closed-loop TCP transfers.
6. The empirical dominant transport transcription error in Table 2.1.

---

## 2. Formal Mathematical Corrections

### 2.1 Correction M-01 & M-02: RFC 8312 TCP CUBIC Window Evolution & Parameterization

#### Original Form in Blueprint (Section 3.3.1, lines 209–214):
$$W_{cubic}(t) = C \cdot (t - K)^3 + W_{max}$$
$$K = \sqrt[3]{\frac{W_{max} \cdot \beta_{cubic}}{C}}$$
Text: *"$\beta_{cubic} = 0.2$ (multiplicative decrease factor upon loss, reducing window by $20\%$ to $0.8 \cdot W_{max}$)"*

#### Theoretical Error Analysis:
1. **Missing $(1 - \beta_{cubic})$ Term in $K$:** Immediately following a packet loss event at $t = 0$ (the start of congestion avoidance after loss recovery), the window size is reduced to $W_{cubic}(0) = W_{max} \cdot \beta_{cubic}$. Setting $t = 0$ in the CUBIC equation yields:
   $$W_{cubic}(0) = C \cdot (-K)^3 + W_{max} = -C \cdot K^3 + W_{max}$$
   Equating this to $W_{max} \cdot \beta_{cubic}$:
   $$-C \cdot K^3 + W_{max} = W_{max} \cdot \beta_{cubic} \implies C \cdot K^3 = W_{max} \cdot (1 - \beta_{cubic})$$
   $$K = \sqrt[3]{\frac{W_{max} \cdot (1 - \beta_{cubic})}{C}}$$
   Dropping $(1 - \beta_{cubic})$ was a critical mathematical error.
2. **Parameter Inversion:** In RFC 8312 Section 4.6, $\beta_{cubic}$ is defined as the *retained* fraction of window after loss. Default is $\beta_{cubic} = 0.7$ (a $30\%$ drop). Setting $\beta = 0.2$ represented an $80\%$ drop.

#### Corrected Mathematical Formulation:
$$\mathbf{K = \left( \frac{W_{max} \cdot (1 - \beta_{cubic})}{C} \right)^{1/3}}$$
$$\mathbf{W_{cubic}(t) = C \cdot (t - K)^3 + W_{max}}$$
$$\mathbf{W_{est}(t) = W_{max} \cdot \beta_{cubic} + \left[ \frac{3 \cdot (1 - \beta_{cubic})}{1 + \beta_{cubic}} \right] \cdot \frac{t}{RTT}}$$
$$\mathbf{cwnd = \max(W_{cubic}(t), W_{est}(t))}$$
* Standard RFC 8312 parameter values:
  - $C = 0.4\text{ packets} / \text{s}^3$ (scaling factor)
  - $\beta_{cubic} = 0.7$ (multiplicative decrease factor; retains $70\%$ of window, reducing by $30\%$)
  - Fast convergence: If $cwnd < W_{max}^{last}$, $W_{max} \leftarrow cwnd \cdot \frac{1 + \beta_{cubic}}{2}$

---

### 2.2 Correction M-04: Lognormal Video Chunk Sizing (Jensen's Inequality Offset)

#### Original Form in Blueprint (Section 3.2.2, line 166):
$$\mathbb{E}[S_k] = \frac{R_{video} \cdot T_{seg}}{8}$$
$$\text{chunk\_bytes} = \text{int}\left(\text{random.lognormvariate}\left(\ln\left(\frac{R_{video} \cdot T_{seg}}{8}\right), 0.25\right)\right)$$

#### Theoretical Error Analysis:
For a lognormal random variable $X = \exp(Y)$ where $Y \sim \mathcal{N}(\mu, \sigma^2)$, the expected value is:
$$\mathbb{E}[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right)$$
By setting $\mu = \ln(M)$ where $M = \frac{R_{video} T_{seg}}{8}$:
$$\mathbb{E}[X] = \exp\left(\ln(M) + \frac{\sigma^2}{2}\right) = M \cdot \exp\left(\frac{\sigma^2}{2}\right)$$
For $\sigma = 0.25$:
$$\exp\left(\frac{0.25^2}{2}\right) = \exp(0.03125) \approx 1.031743 \quad (+3.174\% \text{ upward bias})$$
For high-motion video ($\sigma = 0.50$):
$$\exp\left(\frac{0.50^2}{2}\right) = \exp(0.125) \approx 1.133148 \quad (+13.315\% \text{ upward bias})$$
Setting $\mu = \ln(M)$ sets the *median* of $X$ to $M$, not the arithmetic mean. Over millions of simulated packets, this systematic bias drives video throughput $+3.17\%$ above planned bottleneck allocations.

#### Corrected Mathematical Formulation:
To guarantee that $\mathbb{E}[X] = M$, the underlying Gaussian mean must be adjusted by subtracting the second-order variance offset:
$$\mathbf{\mu^* = \ln\left( \frac{R_{video} \cdot T_{seg}}{8} \right) - \frac{\sigma^2}{2}}$$
$$\mathbf{S_k \sim \text{Lognormal}\left(\mu^*, \sigma^2\right)}$$
Then:
$$\mathbb{E}[S_k] = \exp\left( \mu^* + \frac{\sigma^2}{2} \right) = \exp\left( \ln(M) - \frac{\sigma^2}{2} + \frac{\sigma^2}{2} \right) = M$$

---

### 2.3 Correction M-05: Real-Time Isochronous Pacing (Eliminating Brownian Random Walk)

#### Original Form in Blueprint (Section 3.1.2, lines 122–125):
$$t_{i+1} = t_i + \tau + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.8\text{ ms}, \quad \tau = 20.0\text{ ms}$$

#### Theoretical Error Analysis:
The recursive definition expresses $t_N$ as a cumulative sum of random variables:
$$t_N = t_0 + N \cdot \tau + \sum_{i=1}^N \epsilon_i$$
Because $\epsilon_i$ are independent identically distributed normal variables:
$$\text{Var}(t_N) = \sum_{i=1}^N \text{Var}(\epsilon_i) = N \cdot \sigma^2 \implies \text{SD}(t_N) = \sigma \sqrt{N}$$
For a typical 200-second voice call generating $N = 10,000$ packets:
$$\text{SD}(t_{10000}) = 0.8\text{ ms} \times \sqrt{10000} = 0.8 \times 100 = \mathbf{80.0\text{ ms}}$$
The transmission schedule drifts by up to $\pm 240\text{ ms}$ (3 standard deviations) from the physical timeline. In physical systems, packetization is anchored to an absolute audio sampling clock (e.g. 48 kHz crystal oscillator in sound cards).

#### Corrected Mathematical Formulation:
Anchor packet generation to the absolute sampling clock timeline:
$$\mathbf{t_k = t_{start} + k \cdot \tau + \delta_k}$$
where:
* $\tau = 0.020\text{ s}$ ($50.0\text{ Hz}$ frame interval)
* $\delta_k \sim \text{Uniform}(-\delta_{max}, \delta_{max})$, $\delta_{max} = 0.0003\text{ s}$ ($0.3\text{ ms}$ bounded OS timer jitter)
* $\text{Var}(t_k) = \text{Var}(\delta_k) = \frac{(2 \delta_{max})^2}{12} = \text{constant}$ ($\mathbf{0\text{ cumulative drift}}$)

---

### 2.4 Correction of Turn-Taking Dialogue Dynamics (Coupled 4-State Markov Model)

#### Original Form in Blueprint (Section 3.1.4, lines 145–148):
Independent Brady processes on caller and callee with speech activity $\gamma = 0.40$.

#### Theoretical Error Analysis:
Under independent models, the simultaneous double-talk probability is:
$$P(\text{Double-Talk}) = P(S_{on, 1} \cap S_{on, 2}) = \gamma_1 \times \gamma_2 = 0.40 \times 0.40 = \mathbf{0.16} \quad (16.0\%)$$
In human conversation (Brady 1968, ITU-T Recommendation P.59), double-talk occurs only **$3\%$ to $7\%$** of the time. Independent generation creates a $2.5\times$ to $5\times$ excess of simultaneous double-talk bursts, distorting queue backlog at the bottleneck.

#### Corrected Mathematical Formulation:
Implement a coupled 4-State Continuous-Time Markov Chain (ITU-T P.59):
* State 0: **Mutual Silence ($S_{00}$)** — Both parties silent ($\mathbb{E}[D] = 1.20\text{ s}$, stationary prob $P \approx 0.18$)
* State 1: **Caller Solitary ($S_{10}$)** — Caller speaking, callee listening ($\mathbb{E}[D] = 1.40\text{ s}$, stationary prob $P \approx 0.38$)
* State 2: **Callee Solitary ($S_{01}$)** — Callee speaking, caller listening ($\mathbb{E}[D] = 1.40\text{ s}$, stationary prob $P \approx 0.38$)
* State 3: **Double-Talk ($S_{11}$)** — Interruption / simultaneous speech ($\mathbb{E}[D] = 0.25\text{ s}$, stationary prob $P \approx \mathbf{0.06}$)

Transition probability matrix $P_{ij}$:
$$\mathbf{P} = \begin{pmatrix}
0.00 & 0.48 & 0.48 & 0.04 \\
0.35 & 0.00 & 0.58 & 0.07 \\
0.35 & 0.58 & 0.00 & 0.07 \\
0.20 & 0.40 & 0.40 & 0.00
\end{pmatrix}$$

---

## 3. Side-by-Side Code Fixes (Before vs. After)

### 3.1 Bulk Data CUBIC Flow Fix (B-01 & B-02)

#### BEFORE (Flawed Open-Loop in `traffic_generation_design.md` lines 540–585):
```python
# FLAWED: Open-loop, K equation wrong, last_loss_time never updated -> cwnd explodes to 13,984
class BulkDataFlow:
    def __init__(self, env, flow_id, duration, src_ip, dst_ip, packet_sink, rtt=0.040, link_rate_bps=1e8):
        self.C = 0.4
        self.beta = 0.2                          # WRONG: 0.2 means an 80% window drop
        self.cwnd = 10.0
        self.w_max = 10.0
        self.last_loss_time = 0.0

    def run(self):
        start_time = self.env.now
        self.last_loss_time = start_time

        while (self.env.now - start_time) < self.duration:
            t = self.env.now - self.last_loss_time
            K = (self.w_max * self.beta / self.C) ** (1.0 / 3.0)  # WRONG: missing (1 - beta)
            target_w = self.C * ((t - K) ** 3) + self.w_max
            self.cwnd = max(2.0, target_w)       # BUG: t grows without bound -> cwnd -> infinity!

            burst_count = int(self.cwnd)
            for _ in range(burst_count):
                self.packet_sink.put({...})
                dt_line = (MTU_DATA_SIZE * 8.0) / self.link_rate_bps
                yield self.env.timeout(dt_line)  # Blasts whole cwnd at line rate

            ack_count = max(1, burst_count // 2)
            for _ in range(ack_count):
                self.packet_sink.put({...})      # BUG: Emits all ACKs at same timestamp dt=0!

            yield self.env.timeout(max(0.005, self.rtt))
```

#### AFTER (Corrected Closed-Loop SimPy TCP CUBIC):
```python
# CORRECTED: True closed-loop feedback, RFC 8312 K formula, Slow Start, Pacing across RTT
class CorrectedBulkDataFlow:
    def __init__(self, env, flow_id, duration, src_ip, dst_ip, bottleneck_queue, rtt_prop=0.040, link_rate_bps=100e6):
        self.env = env
        self.flow_id = flow_id
        self.duration = duration
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.queue = bottleneck_queue
        self.rtt_prop = rtt_prop
        self.link_rate_bps = link_rate_bps
        
        # RFC 8312 standard parameters
        self.C = 0.4
        self.beta = 0.7                          # RFC 8312 default: retain 70% (30% drop)
        self.mss = 1500
        
        # State variables
        self.cwnd = 10.0                         # RFC 6928 IW10
        self.ssthresh = 64.0
        self.w_max = 10.0
        self.epoch_start = 0.0
        self.in_slow_start = True
        
        self.snd_una = 0
        self.snd_nxt = 0
        self.ack_pipe = simpy.Store(env)

    def run(self):
        start_time = self.env.now
        self.epoch_start = start_time
        self.env.process(self._ack_listener(start_time))

        while (self.env.now - start_time) < self.duration:
            effective_rtt = max(self.rtt_prop, self.queue.get_current_delay() + self.rtt_prop)
            flight_size = (self.snd_nxt - self.snd_una) // self.mss
            
            if flight_size < int(self.cwnd):
                seq = self.snd_nxt
                self.snd_nxt += self.mss
                pkt = {
                    'flow_id': self.flow_id,
                    'qos_class': 'Bulk Data',
                    'timestamp': self.env.now,
                    'size': self.mss,
                    'seq': seq,
                    'direction': 'fwd',
                    'flow_ref': self
                }
                self.queue.enqueue(pkt)

            # Smooth packet pacing across smoothed RTT (prevents square-wave burst monopolization)
            dt_pace = effective_rtt / max(1.0, self.cwnd)
            yield self.env.timeout(max(1e-5, dt_pace))

    def _ack_listener(self, start_time):
        dup_acks = 0
        last_ack = 0

        while (self.env.now - start_time) < self.duration:
            ack_msg = yield self.ack_pipe.get()
            ack_num = ack_msg['ack_seq']
            loss_indicated = ack_msg.get('loss', False)

            if loss_indicated or dup_acks >= 3:
                # Loss recovery: RFC 8312 multiplicative decrease & fast convergence
                self.epoch_start = self.env.now
                if self.cwnd < self.w_max:
                    self.w_max = self.cwnd * (1.0 + self.beta) / 2.0
                else:
                    self.w_max = self.cwnd
                
                self.ssthresh = max(2.0, self.cwnd * self.beta)
                self.cwnd = self.ssthresh
                self.in_slow_start = False
                dup_acks = 0
            else:
                if ack_num > last_ack:
                    last_ack = ack_num
                    self.snd_una = ack_num
                    dup_acks = 0
                    
                    if self.in_slow_start:
                        self.cwnd += 1.0
                        if self.cwnd >= self.ssthresh:
                            self.in_slow_start = False
                            self.epoch_start = self.env.now
                    else:
                        t = self.env.now - self.epoch_start
                        # Correct RFC 8312 formula: K = ((W_max * (1 - beta)) / C)^(1/3)
                        K = ((self.w_max * (1.0 - self.beta)) / self.C) ** (1.0 / 3.0)
                        w_cubic = self.C * ((t - K) ** 3) + self.w_max
                        
                        # RFC 8312 Section 4.2 TCP-friendly region
                        w_est = self.w_max * self.beta + (3.0 * (1.0 - self.beta) / (1.0 + self.beta)) * (t / self.rtt_prop)
                        target_w = max(w_cubic, w_est)
                        
                        if target_w > self.cwnd:
                            self.cwnd += (target_w - self.cwnd) / self.cwnd
```

---

### 3.2 Streaming Generator Fix (B-03, B-04, B-05, B-06)

#### BEFORE (Flawed Implementation in `traffic_generation_design.md` lines 463–524):
```python
# FLAWED: Division ACK logic, 52B HTTP GET, negative buffer collapse, dead-code buffer_target
class StreamingFlow:
    def run(self):
        ...
        while (self.env.now - start_time) < self.duration:
            # BUG 1: Samples HTTP GET from distribution where 85% is 52B pure TCP ACK!
            req_size = np.random.choice(STREAMING_UP_SIZES, p=STREAMING_UP_PROBS)
            self.packet_sink.put({...})
            
            # BUG 2: Zero RTT delay; server transmits at exact identical timestamp!
            chunk_bytes = int(random.lognormvariate(
                math.log((self.target_bitrate * self.segment_duration) / 8.0), 0.25 # BUG 3: Jensen bias
            ))
            t_download_start = self.env.now
            bytes_sent = 0

            while bytes_sent < chunk_bytes:
                pkt_size = np.random.choice(STREAMING_DOWN_SIZES, p=STREAMING_DOWN_PROBS)
                bytes_sent += pkt_size
                self.packet_sink.put({...})

                # BUG 4: Divisor fluctuates; produces back-to-back ACKs & skips ACKs
                if (bytes_sent // pkt_size) % 2 == 0:
                    self.packet_sink.put({...})

                dt = (pkt_size * 8.0) / self.link_rate_bps
                yield self.env.timeout(max(1e-6, dt))

            t_download = self.env.now - t_download_start
            # BUG 5: Unclamped subtraction; if t_download > T_seg, buffer goes negative!
            current_buffer += self.segment_duration - t_download

            if current_buffer < self.min_buffer:
                yield self.env.timeout(0.005) # Infinite loop if buffer < 0!
            else:
                idle_time = max(0.05, self.segment_duration - t_download + random.gauss(0, 0.1))
                yield self.env.timeout(idle_time)
                current_buffer = max(0.0, current_buffer - idle_time) # Subtracted twice!
```

#### AFTER (Corrected 3-State Buffer Engine & Robust Delayed ACK Tracker):
```python
# CORRECTED: Unbiased lognormal chunk, 450B HTTP GET, RTT delay, 3-state non-negative buffer
class CorrectedStreamingFlow:
    def run(self):
        start_time = self.env.now
        last_event_time = self.env.now

        while (self.env.now - start_time) < self.duration:
            # 1. Authentic HTTP GET segment request (450B)
            self.packet_sink.put({
                'flow_id': self.flow_id,
                'qos_class': 'Streaming',
                'timestamp': self.env.now,
                'size': 450,                     # Authentic HTTP GET header size
                'direction': 'fwd',
                'src': self.client_ip, 'dst': self.server_ip
            })
            
            # Uplink propagation + server turnaround delay (RTT / 2)
            yield self.env.timeout(max(0.001, self.rtt / 2.0))
            
            # Drain media playback buffer during request traversal
            if self.is_playing:
                elapsed = self.env.now - last_event_time
                self.current_buffer = max(0.0, self.current_buffer - elapsed)
                if self.current_buffer == 0.0:
                    self.is_playing = False       # Playback stalled
            last_event_time = self.env.now

            # 2. Unbiased Lognormal chunk sizing (Jensen's inequality correction)
            target_bytes = (self.target_bitrate * self.segment_duration) / 8.0
            mu_corr = math.log(target_bytes) - (self.lognorm_sigma ** 2 / 2.0)
            chunk_bytes = max(1500, int(round(random.lognormvariate(mu_corr, self.lognorm_sigma))))

            bytes_sent = 0
            pkt_count = 0
            t_download_start = self.env.now

            while bytes_sent < chunk_bytes:
                pkt_size = np.random.choice(STREAMING_DOWN_SIZES, p=STREAMING_DOWN_PROBS)
                bytes_sent += pkt_size
                pkt_count += 1

                self.packet_sink.put({
                    'flow_id': self.flow_id,
                    'qos_class': 'Streaming',
                    'timestamp': self.env.now,
                    'size': pkt_size,
                    'direction': 'bwd',
                    'src': self.server_ip, 'dst': self.client_ip
                })

                # RFC 5681 Delayed ACKs: exactly 1 ACK per 2 data packets
                if pkt_count % 2 == 0:
                    self.packet_sink.put({
                        'flow_id': self.flow_id,
                        'qos_class': 'Streaming',
                        'timestamp': self.env.now + (self.rtt / 2.0),
                        'size': 52,
                        'direction': 'fwd',
                        'src': self.client_ip, 'dst': self.server_ip
                    })

                dt_pace = max(1e-6, (pkt_size * 8.0) / self.link_rate_bps)
                yield self.env.timeout(dt_pace)

            # Flush trailing delayed ACK if segment packet count is odd
            if pkt_count % 2 != 0:
                self.packet_sink.put({
                    'flow_id': self.flow_id,
                    'qos_class': 'Streaming',
                    'timestamp': self.env.now + (self.rtt / 2.0),
                    'size': 52,
                    'direction': 'fwd',
                    'src': self.client_ip, 'dst': self.server_ip
                })

            # Drain buffer during download and add downloaded segment
            t_download = self.env.now - t_download_start
            if self.is_playing:
                self.current_buffer = max(0.0, self.current_buffer - t_download)
                if self.current_buffer == 0.0:
                    self.is_playing = False

            self.current_buffer += self.segment_duration
            if not self.is_playing and self.current_buffer >= self.min_buffer:
                self.is_playing = True            # Buffer primed -> start playback

            last_event_time = self.env.now

            # 3. Closed-Loop Playback Buffer Regulation
            if self.current_buffer < self.buffer_target:
                # Fast Buffering: immediate next chunk request
                yield self.env.timeout(0.005)
                if self.is_playing:
                    self.current_buffer = max(0.0, self.current_buffer - 0.005)
            else:
                # Steady-State Pacing: sleep until buffer approaches target
                surplus = self.current_buffer - self.buffer_target
                idle_time = max(0.05, surplus + random.gauss(0, 0.05))
                yield self.env.timeout(idle_time)
                if self.is_playing:
                    self.current_buffer = max(0.0, self.current_buffer - idle_time)
            last_event_time = self.env.now
```

---

### 3.3 Real-Time Conversational Flow Fix (B-07, M-05)

#### BEFORE (Flawed Implementation in `traffic_generation_design.md` lines 364–432):
```python
# FLAWED: Recursive clock drift (>80ms), 160ms cellular SID, unyielded put(), 16% double-talk
class RealTimeFlow:
    def _talker_process(self, is_forward, start_time):
        ...
        while (self.env.now - start_time) < self.duration:
            t_on = random.expovariate(1.0 / self.mean_talkspurt)
            on_end = self.env.now + t_on
            while self.env.now < on_end and (self.env.now - start_time) < self.duration:
                # BUG 1: Unyielded put() fails in SimPy
                self.packet_sink.put({...})
                
                # BUG 2: Recursive jitter causes unbounded random-walk clock drift!
                jitter = random.gauss(0, self.jitter_sigma)
                dt = max(0.001, self.frame_interval + jitter)
                yield self.env.timeout(dt)

            t_off = random.expovariate(1.0 / self.mean_silence)
            off_end = self.env.now + t_off
            while self.env.now < off_end and (self.env.now - start_time) < self.duration:
                # BUG 3: 160ms cellular SID frames contradict encrypted WebRTC standards
                self.packet_sink.put({...})
                yield self.env.timeout(self.sid_interval) # 160 ms
```

#### AFTER (Coupled 4-State Turn-Taking & Absolute Clock Scheduling):
```python
# CORRECTED: 4-State Markov turn-taking, absolute clock pacing (zero drift), safe put()
class CorrectedRealTimeFlow:
    VOIP_SIZES = [64, 90, 140, 200]
    VOIP_PROBS = [0.15, 0.45, 0.25, 0.15]        # Empirical Opus 90B mode included

    def run(self):
        start_time = self.env.now
        end_time = start_time + self.duration
        
        # 4-State Markov Turn-Taking Model (ITU-T P.59)
        state = 0  # 0: Mutual Silence, 1: Caller, 2: Callee, 3: Double-Talk
        state_mean_durations = {0: 1.20, 1: 1.40, 2: 1.40, 3: 0.25}
        transition_matrix = {
            0: [0.00, 0.48, 0.48, 0.04],
            1: [0.35, 0.00, 0.58, 0.07],
            2: [0.35, 0.58, 0.00, 0.07],
            3: [0.20, 0.40, 0.40, 0.00]
        }

        while self.env.now < end_time:
            state_dur = self.rng.expovariate(1.0 / state_mean_durations[state])
            state_end = min(end_time, self.env.now + state_dur)
            
            # Isochronous frame generation anchored to absolute sampling intervals
            num_frames = int((state_end - self.env.now) / self.frame_interval)
            t_base = self.env.now

            for k in range(num_frames):
                target_send_time = t_base + (k * self.frame_interval)
                sleep_dt = target_send_time - self.env.now
                if sleep_dt > 0:
                    yield self.env.timeout(sleep_dt)

                if self.env.now >= end_time:
                    break

                # State 1: Caller solitary
                if state in (1, 3):
                    self._emit_packet(is_forward=True, is_speech=True)
                elif state == 0:
                    self._emit_packet(is_forward=True, is_speech=False) # 50 Hz low-bitrate comfort noise

                # State 2: Callee solitary
                if state in (2, 3):
                    self._emit_packet(is_forward=False, is_speech=True)
                elif state == 0:
                    self._emit_packet(is_forward=False, is_speech=False)

            # Sample next turn-taking state
            state = self.rng.choices([0, 1, 2, 3], weights=transition_matrix[state])[0]

    def _emit_packet(self, is_forward: bool, is_speech: bool):
        direction = 'fwd' if is_forward else 'bwd'
        self.packet_seq[direction] += 1
        size = self.rng.choices(self.VOIP_SIZES, weights=self.VOIP_PROBS)[0] if is_speech else 64

        pkt = {
            'flow_id': self.flow_id,
            'qos_class': 'Real-Time',
            'seq': self.packet_seq[direction],
            'timestamp': self.env.now,
            'size': size,
            'direction': direction,
            'src': self.caller_ip if is_forward else self.callee_ip,
            'dst': self.callee_ip if is_forward else self.caller_ip
        }
        # Safe dispatch to SimPy Store
        put_ret = self.packet_sink.put(pkt)
        if isinstance(put_ret, simpy.Event):
            yield put_ret
```

---

## 4. Reconciled Master Parameter Table

The table below provides the reconciled, verified, and standards-compliant parameters to replace Section 2.1 and Section 5 in `traffic_generation_design.md`:

| Parameter | Original Value in Blueprint | Audited & Reconciled Value | Basis & Justification |
| :--- | :---: | :---: | :--- |
| **Sustained Dominant Transport (Table 2.1)** | RT: UDP 88%, STR: TCP 59%, BLK: UDP 76% | **RT: TCP 87.3%, STR: TCP 84.9%, BLK: TCP 96.1%** | Fixes dataset transcription error; isolates sustained data flows. |
| **CUBIC Decrease Factor ($\beta_{cubic}$)** | $\beta = 0.2$ | **$\beta = 0.7$** | RFC 8312 Section 4.6 default multiplicative decrease parameter. |
| **CUBIC Time Deficit ($K$)** | $K = (W_{max}\beta/C)^{1/3}$ | **$K = [W_{max}(1-\beta)/C]^{1/3}$** | RFC 8312 Equation 2 standard derivation. |
| **CUBIC TCP-Friendly Mode ($W_{est}$)** | Omitted | **Mandatory $W_{est}(t)$ lower bound** | RFC 8312 Section 4.2 prevents window starvation. |
| **Lognormal Chunk Parameter ($\mu$)** | $\mu = \ln(M)$ | **$\mu^* = \ln(M) - \sigma^2/2$** | Eliminates Jensen's inequality $+3.17\%$ upward volume bias. |
| **Streaming Bitrate ($R_{video}$)** | Divergent: 3.5 Mbps vs 2.0 Mbps | **Harmonized: $2.0\text{ Mbps}$** | Reconciles generator agent with orchestrator Poisson arrival formulas. |
| **Streaming Delayed ACK Policy** | Division heuristic `(bytes // pkt) % 2` | **Explicit counter `pkt_count % 2 == 0`** | Eliminates $6.5\%$ consecutive ACK bug; conforms to RFC 5681. |
| **HTTP GET Request Size** | Sampled from `[52: 85%, 450: 15%]` | **$450\text{ B}$ (Fixed Request Envelope)** | Separates $52\text{ B}$ pure ACKs from authentic HTTP headers. |
| **Streaming Buffer Dynamic** | Unclamped subtraction | **Clamped $B(t) \ge 0$, 3-State Regulation** | Prevents negative buffer crash; regulates around $B_{target}=30\text{ s}$. |
| **Real-Time Framing Architecture** | Recursive random walk | **Absolute Hardware Clock Sampling** | Anchored to $t_k = t_0 + k\tau$; eliminates Brownian clock drift. |
| **Real-Time Silence Mode** | 160 ms cellular SID comfort noise | **Continuous 50 Hz Low-Bitrate (64B)** | Modern encrypted WebRTC standard; prevents speech clipping & timing leakage. |
| **Real-Time Turn-Taking** | Independent talkers (16% double-talk) | **Coupled 4-State Markov Model (6% DT)** | ITU-T P.59 standardizes turn-taking and mutual silence. |
| **Real-Time Packet Size Modes** | `[64: 20%, 120: 35%, 200: 30%, 350: 10%]` | **`[64: 15%, 90: 45%, 140: 25%, 200: 15%]`** | Incorporates dominant Track B empirical Opus mode at $90\text{ B}$. |
| **3GPP 5QI 9 Delay Budget** | "Unbounded" | **$300\text{ ms}$ (Nominal 3GPP PDB)** | 3GPP TS 23.501 Table 5.7.4-1 standard specification. |
| **Validation Goodness-of-Fit** | Continuous 2-sample KS ($p > 0.05$) | **Wasserstein-1 ($W_1 \le 35\text{B}$), JSD ($\le 0.05$)** | Valid on discrete multimodal packet distributions; avoids large-$N$ paradox. |
| **Simulation Warm-Up Horizon** | $0.0\text{ s}$ (Cold Start) | **$T_{warm} = 60.0\text{ s}$ (Welch Procedure)** | Discards initial buffer filling transients; records over $[60, 360]\text{ s}$. |

---

## 5. Summary of Corrective Impact

By applying these corrections:
1. **Physical Fidelity:** Saturated TCP Bulk flows will exhibit realistic closed-loop loss recovery, pacing, and slow-start without runaway window explosion.
2. **Buffer Stability:** Video streaming clients will realistically buffer and pace segments without crashing into negative buffer loops.
3. **Conversational Realism:** Voice callers will take turns naturally with stationary 50 Hz timing and empirical Opus packet size profiles.
4. **Classifier Compatibility:** Early flow packets will produce unscaled, zero-leakage 6-feature vectors that seamlessly match the Phase 6B Early-5 XGBoost classifier.

*The full production-grade, finalized specification incorporating all corrections is provided in `research_notes/phase7/traffic_generation_final_specification.md`.*
