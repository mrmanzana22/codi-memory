---
description: Diagnose codi-memory system health (Sprint 3-4 implementations)
---

# System Diagnostic

Run the system diagnostic script and interpret the results:

## Live Data
- Diagnostic report: !`cd /Users/codi-air/codi-memory && ./venv/bin/python3 scripts/debug_system.py 2>&1`

## Instructions

Analyze the diagnostic report above and provide:

1. **Status summary**: Which components are healthy, which need attention
2. **Dual Process (S4-04)**: Is habit mode firing too much? What's the hit rate?
3. **World Model (S4-05)**: Has it learned anything? Are tick timestamps recent?
4. **GWT Bistability (S3-01)**: Are winning scores showing ignition (>0.9)?
5. **Bridge Edges (S3-05)**: Any cross-topic bridges created?
6. **Active Inference**: Can the module compute EFE and select actions?
7. **Recommendations**: What needs fixing or monitoring

Keep the analysis concise and actionable. Flag any ERROR or NO_DATA sections as priorities.
