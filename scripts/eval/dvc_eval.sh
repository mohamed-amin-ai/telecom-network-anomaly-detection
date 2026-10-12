#!/usr/bin/env bash
# DVC: initial reproduce, no-op reproduce, then change ONE downstream param and see which stages rerun.
set -u
EVAL_DIR=${EVAL_DIR:-eval/core}
mkdir -p "$EVAL_DIR"
dvc init -q 2>&1 || true
dvc config core.analytics false
stages() { grep -E "^Running stage|didn't change, skipping" "$1" | sed -E "s/^Running stage '([^']+)'.*/ran:\1/; s/^Stage '([^']+)' didn't change, skipping.*/skipped:\1/" | tr '\n' ' '; }
t0=$(date +%s); dvc repro > /tmp/dvc1.log 2>&1; rc1=$?; t1=$(date +%s)
cat /tmp/dvc1.log | tail -n 15
cp models/metadata.json /tmp/meta_base.json 2>/dev/null
dvc repro > /tmp/dvc2.log 2>&1; rc2=$?
sed -i 's/validation_recall_target: 0\.945/validation_recall_target: 0.94/' params.yaml
grep -qE 'validation_recall_target: 0\.94$' params.yaml || { echo 'dvc_eval: param edit did not apply'; exit 1; }
dvc params diff --json > "$EVAL_DIR/dvc_params_diff.json" 2>/dev/null
t2=$(date +%s); dvc repro > /tmp/dvc3.log 2>&1; rc3=$?; t3=$(date +%s)
cp models/metadata.json /tmp/meta_changed.json 2>/dev/null
git checkout -- params.yaml
dvc repro > /tmp/dvc4.log 2>&1; rc4=$?
dvc dag > "$EVAL_DIR/dvc_dag.txt" 2>&1
python3 - "$EVAL_DIR" "$rc1" "$rc2" "$rc3" "$rc4" "$((t1-t0))" "$((t3-t2))" "$(stages /tmp/dvc1.log)" "$(stages /tmp/dvc2.log)" "$(stages /tmp/dvc3.log)" "$(stages /tmp/dvc4.log)" <<'PY'
import json, sys
d, r1, r2, r3, r4, s1, s3, st1, st2, st3, st4 = sys.argv[1:12]
def m(p):
    try: return json.load(open(p))["metrics"]
    except Exception: return None
json.dump({
    "first_repro": {"exit": int(r1), "seconds": int(s1), "stages": st1.split()},
    "second_repro_no_changes": {"exit": int(r2), "stages": st2.split()},
    "after_changing_validation_recall_target_0.945_to_0.94": {"exit": int(r3), "seconds": int(s3), "stages": st3.split(),
        "metrics_before": m("/tmp/meta_base.json"), "metrics_after": m("/tmp/meta_changed.json")},
    "restore_params_repro": {"exit": int(r4), "stages": st4.split()},
}, open(f"{d}/dvc.json", "w"), indent=2)
PY
cat "$EVAL_DIR/dvc.json"
[ "$rc1" = 0 ] && [ "$rc2" = 0 ] && [ "$rc3" = 0 ] && [ "$rc4" = 0 ]
