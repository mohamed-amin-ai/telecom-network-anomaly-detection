#!/usr/bin/env bash
# Fire a fake critical alert straight into Alertmanager to see the email path work end to end,
# without having to break the API or wait for drift. Open http://localhost:8025 (Mailpit) afterwards.
# Usage: scripts/send_test_alert.sh [alertmanager_url]
set -euo pipefail
AM="${1:-http://localhost:9093}"
now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
curl -fsS -X POST "$AM/api/v2/alerts" -H 'Content-Type: application/json' -d "[
  {\"labels\": {\"alertname\": \"AnomalyApiDown\", \"severity\": \"critical\", \"job\": \"telecom-anomaly-api\"},
   \"annotations\": {\"summary\": \"TEST: anomaly API is down\", \"description\": \"Synthetic alert from scripts/send_test_alert.sh.\"},
   \"startsAt\": \"$now\"}
]"
echo "Alert sent. Grouped email arrives in Mailpit after about 10 s: http://localhost:8025"
