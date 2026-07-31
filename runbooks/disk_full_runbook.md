# Runbook: Disk Space Critical (>90% Usage)

## Summary
Triggered when disk/volume usage on a host or mount crosses 90%.
If unaddressed, this can cause service crashes, failed writes,
database corruption, or inability to log — so it needs a fast response.

## Severity
**High** — escalate if usage exceeds 95% or usage is still climbing
after Step 3.

## Detection / Symptoms
- Monitoring alert: "Disk usage > 90%" on host or mount
- Application errors: "No space left on device"
- Failed deployments, failed log writes, database write errors
- `df -h` shows a filesystem near 100%

## Diagnosis Steps

1. **Confirm the alert**
   ```
   df -h
   ```
   Identify which mount/partition is actually full (not just root `/`).

2. **Find what's consuming space**
   ```
   du -sh /var/log/* 2>/dev/null | sort -rh | head -10
   du -sh /tmp/* 2>/dev/null | sort -rh | head -10
   du -sh /home/*/* 2>/dev/null | sort -rh | head -10
   ```
   Common offenders: log files, core dumps, old deployment artifacts,
   docker images/containers, temp files from failed jobs.

3. **Check for deleted-but-open files holding space**
   ```
   lsof +L1
   ```
   A process can hold disk space for a deleted file until it's restarted.

## Remediation Steps

4. **Clear safe-to-delete files first**
   - Rotate/compress/delete old logs:
     ```
     find /var/log -name "*.log" -mtime +7 -exec gzip {} \;
     find /var/log -name "*.gz" -mtime +30 -delete
     ```
   - Clear temp files:
     ```
     find /tmp -type f -mtime +1 -delete
     ```
   - Clean unused Docker resources (if applicable):
     ```
     docker system prune -af
     ```

5. **If a specific service is the culprit**, restart it after clearing
   its files so it releases any file handles on deleted data:
   ```
   systemctl restart <service-name>
   ```

6. **Re-check usage**
   ```
   df -h
   ```
   Confirm usage has dropped back under threshold (<80% is a safe target).

7. **If space still critical after steps 4–6**:
   - Escalate to on-call lead
   - Consider expanding the volume (cloud disk resize) as a stopgap
   - Do NOT delete files you don't recognize — flag to the service owner

## Verification
- `df -h` shows usage back under 80%
- Affected service is writing/logging normally again
- No new "No space left on device" errors in application logs

## Post-Incident
- Log root cause (e.g., "log rotation misconfigured," "runaway job writing temp files")
- File a follow-up ticket for permanent fix (log rotation policy, disk autoscaling, alert threshold tuning)
- Update this runbook if a new root cause type was discovered

## Escalation Contacts
- Primary on-call: _[fill in]_
- Infra/Platform team: _[fill in]_
