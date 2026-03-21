# Troubleshooting Guide

## 1. Port 9100 Already in Use (Prometheus)

**Symptom:** `[warn] PrometheusExporter failed to bind on port 9100` at startup.

**Cause:** Another process (node_exporter, another engine instance) is listening on port 9100.

**Fix:** Find the conflicting process with `ss -tlnp | grep 9100` (Linux) or `netstat -ano | findstr 9100` (Windows). Either stop it or change the Prometheus port by passing `--stats-port N` on the command line, or set `metrics.stats_port: N` in `config.yaml` (hot-reloadable).

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

---

## 9. Config Values Look Wrong at Runtime

**Symptom:** The engine behaves differently than expected (wrong thresholds, unexpected port, wrong log format) and you are unsure which config file was loaded or which environment variables took effect.

**Fix:** Run the engine with `--dump-config` to print the complete effective configuration and exit immediately — no pipeline is started:

```
./LLMTokenStreamQuantEngine --config config.yaml --dump-config
```

This prints every subsystem key/value pair in `key = value` format after YAML loading, environment variable overrides, and validation. If a field is wrong, check whether an `LLMQUANT_*` environment variable is overriding it (`env | grep LLMQUANT`).

---

## 10. Console Output Too Noisy in Production (Daemon Mode)

**Symptom:** The live stats bar and per-signal console output pollute your log aggregator when the engine is run as a daemon or under systemd.

**Fix:** Pass `--quiet` on the command line:

```
./LLMTokenStreamQuantEngine --config config.yaml --quiet
```

In quiet mode all signal-level console output and the stats bar are suppressed. Structured log entries continue to be written to the file sink (configured by `logging.log_file_path`) and the Prometheus `/metrics` endpoint remains fully active. This is the recommended mode for production deployments where logs are collected by a sidecar.

---

## 11. Token Dictionary Inspection (Coverage / Weight Debugging)

**Symptom:** Signals are weaker than expected and you suspect the loaded dictionary does not cover the tokens your LLM produces.

**Fix — list all tokens to stdout:**

```
./LLMTokenStreamQuantEngine --config config.yaml --list-tokens
```

Prints a tab-separated table (token, sentiment, confidence, volatility, bias) for every entry in the loaded dictionary and exits.

**Fix — export dictionary to a TSV file:**

```
./LLMTokenStreamQuantEngine --config config.yaml --export-dict weights.tsv
```

Writes the same table to `weights.tsv` (no header row) for further analysis in a spreadsheet or script. Use this to audit coverage, identify zero-weight tokens, or build a custom dictionary.
