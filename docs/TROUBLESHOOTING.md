# Troubleshooting Guide

## 1. Port 9100 Already in Use (Prometheus)

**Symptom:** `[warn] PrometheusExporter failed to bind on port 9100` at startup.

**Cause:** Another process (node_exporter, another engine instance) is listening on port 9100.

**Fix:** Find the conflicting process with `ss -tlnp | grep 9100` (Linux) or `netstat -ano | findstr 9100` (Windows). Either stop it or change the Prometheus port by modifying `PrometheusExporter::Config::port` in `src/main.cpp` before constructing the exporter.

---

## 2. Config Reload Not Working (Hot-Reload)

**Symptom:** Editing `config.yaml` at runtime produces no `[config] Hot-reloaded` message.

**Cause:** Either (a) the file path passed to `start_watching()` does not match the file being edited, or (b) the process lacks read permission on the file.

**Fix:** Check that the path logged at startup matches the file you are editing. On Linux, verify `ls -l config.yaml` shows read permission for the engine user. On Windows, ensure the file is not locked by another editor. If `start_watching()` returned false (check logs for `Config: failed to start hot-reload watcher thread`), the watcher thread failed to launch — restart the engine.

---

## 3. Redis Connection Dropped (Dedup Fallback)

**Symptom:** Logs show Redis disconnection; the engine continues running.

**Behavior:** The engine falls back transparently to the in-process deduplicator. Dedup effectiveness is unchanged for tokens within a single process. Cross-process dedup (e.g. multiple engine instances sharing a Redis namespace) will not work until Redis is restored.

**Fix:** Restore the Redis instance. The engine will attempt reconnect on the next `check_and_register` call. Monitor `llmquant_dedup_redis_connected` and `llmquant_dedup_redis_reconnect_attempts_total` in the Prometheus metrics.

---

## 4. TLS Handshake Failure (OpenAI Connection)

**Symptom:** Log shows `[stream] SSL_connect failed: ...` and the engine retries indefinitely.

**Cause:** (a) System CA bundle is missing or stale, (b) clock skew causes certificate validation to fail, (c) API key is invalid (HTTP 401 after handshake).

**Fix:**
- Linux: run `update-ca-certificates`.
- Windows: the engine loads the Windows ROOT certificate store automatically — ensure it is up to date via Windows Update.
- Check system clock accuracy (`date` / `w32tm /query /status`).
- Verify the API key with `curl -s -H "Authorization: Bearer $LLMQUANT_API_KEY" https://api.openai.com/v1/models`.

---

## 5. Signals Not Firing (Cooldown / Risk Gates)

**Symptom:** Tokens are processed but no `PASS` signals appear in the output.

**Cause:** Either the cooldown has not elapsed or a risk gate is blocking signals.

**Fix:**
- Check `BLOCK(...)` messages in the output for the reject reason.
- Reduce `signal_cooldown_us` in `config.yaml`.
- Review the risk gate thresholds in `RiskManager::Config` (in `src/main.cpp`).
- For testing, set `risk.disable_*_gate: true` in `config.yaml` to isolate which gate is blocking.
- Monitor `llmquant_signals_blocked_*_total` Prometheus metrics to identify the bottleneck gate.

---

## 6. FIX Session Sequence Gap (Resend Request)

**Symptom:** Log shows `[fix_oms] recv failed, reconnecting` or silent position update gaps.

**Cause:** The FIX acceptor detected a sequence gap and sent a ResendRequest (35=2). The engine handles this by responding with a SequenceReset-GapFill (35=4, 123=Y) — it cannot replay historical messages.

**Fix:** This is handled automatically. If the session does not recover, restart the engine — the Logon message includes `ResetSeqNumFlag=Y` to reset both sides. If gaps recur frequently, investigate network stability between the engine and the OMS.

---

## 7. High Latency P99 Spike (Pressure Backoff)

**Symptom:** P99 latency suddenly jumps; `BKOF` factor in the stats bar is > 1x.

**Cause:** The composite pressure signal (ingestion + semantic + queue) exceeded 0.8, triggering exponential backoff in `LatencyController`.

**Fix:**
- Reduce token ingestion rate or increase `pressure.max_ingestion_rate_tps` in `config.yaml` to recalibrate the ingestion pressure threshold.
- Monitor `llmquant_latency_p99_us` to confirm recovery.
- If semantic pressure is the driver (high token weight variance), consider reducing `trading.bias_sensitivity`.

---

## 8. OMS Adapter Not Connecting (Backoff Explanation)

**Symptom:** `RestOmsAdapter: N consecutive failures, backing off to Xms poll interval` in logs.

**Cause:** The OMS REST endpoint is unreachable. After 3 consecutive failures the adapter doubles its poll interval (up to 30 seconds) to avoid flooding the network with failed requests.

**Fix:** Verify the OMS endpoint is reachable: `curl http://<host>:<port>/positions`. Check firewall rules and the `--oms` flag passed to the engine. Once the endpoint recovers, the adapter resets its poll interval to the configured default and logs `RestOmsAdapter: connection recovered after N failures`.
