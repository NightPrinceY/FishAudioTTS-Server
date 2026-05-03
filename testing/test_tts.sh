#!/bin/bash
# Fish Audio S2-Pro TTS — smoke test
# Sends a short Arabic sentence to the server and saves the output WAV.

set -euo pipefail

SERVER="http://localhost:3008"
OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_WAV="${OUTPUT_DIR}/sample_output.wav"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Fish Audio S2-Pro TTS — Smoke Test"
echo "  Server: ${SERVER}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Health check
echo ""
echo "[1/3] Health check..."
HEALTH=$(curl -sf "${SERVER}/health" 2>&1) || { echo "ERROR: Server not reachable at ${SERVER}"; exit 1; }
echo "      ${HEALTH}"

STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
if [ "$STATUS" != "ready" ]; then
  echo "ERROR: Server status is '${STATUS}', expected 'ready'. Is it still loading?"
  exit 1
fi
echo "      Server is READY"

# 2. TTS request
echo ""
echo "[2/3] Sending TTS request..."
TEXT="مرحباً، هذا اختبار لخدمة تحويل النص إلى كلام."

HTTP_CODE=$(curl -s -o "${OUTPUT_WAV}" -w "%{http_code}" \
  -X POST "${SERVER}/v1/tts" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"${TEXT}\", \"max_new_tokens\": 512}" \
  --max-time 120)

if [ "$HTTP_CODE" != "200" ]; then
  echo "ERROR: HTTP ${HTTP_CODE}"
  cat "${OUTPUT_WAV}" 2>/dev/null || true
  exit 1
fi

# 3. Verify output
echo ""
echo "[3/3] Verifying output..."
FILE_SIZE=$(stat -c%s "${OUTPUT_WAV}" 2>/dev/null || echo 0)
if [ "$FILE_SIZE" -lt 1000 ]; then
  echo "ERROR: Output file is suspiciously small (${FILE_SIZE} bytes). Generation may have failed."
  exit 1
fi

echo "      Saved: ${OUTPUT_WAV}"
echo "      Size:  ${FILE_SIZE} bytes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  TEST PASSED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
