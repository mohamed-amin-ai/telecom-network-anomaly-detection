#!/usr/bin/env bash
# Run one evaluation step, record pass/fail + duration in $EVAL_DIR/status.json, keep its log,
# surface failures as annotations, and NEVER fail the job (so one broken tool doesn't hide the rest).
#   scripts/eval/ci_step.sh <name> <command...>
name=$1; shift
EVAL_DIR=${EVAL_DIR:-eval/core}
mkdir -p "$EVAL_DIR/logs"
start=$(date +%s)
"$@" > "$EVAL_DIR/logs/$name.log" 2>&1
rc=$?
dur=$(( $(date +%s) - start ))
python3 - "$EVAL_DIR" "$name" "$rc" "$dur" <<'PY'
import json, os, sys
d, name, rc, dur = sys.argv[1:5]
p = os.path.join(d, "status.json")
s = json.load(open(p)) if os.path.exists(p) else {}
s[name] = {"ok": rc == "0", "seconds": int(dur)}
json.dump(s, open(p, "w"), indent=2)
PY
tail -n 400 "$EVAL_DIR/logs/$name.log" > "$EVAL_DIR/logs/$name.tail" && mv "$EVAL_DIR/logs/$name.tail" "$EVAL_DIR/logs/$name.log"
if [ "$rc" -ne 0 ]; then
  echo "::error title=eval step $name failed::$(tail -n 14 "$EVAL_DIR/logs/$name.log" | cut -c1-220 | tr -d '%\r' | tr '\n' '|')"
fi
echo "== $name: rc=$rc in ${dur}s"; tail -n 25 "$EVAL_DIR/logs/$name.log"
exit 0
