#!/usr/bin/env bash
# Decoy-free depth ladder on the 26B-A4B with LM Studio's argument shape.
#   usage: bash tests-data/depth-ladder.sh kvmem|stock [DEPTHS="120000 240000"]
# Each depth carries exactly one unique serial (gen_ladder.py enforces this; an
# earlier revision repeated a filler corpus that contained a second serial and the
# "depth boundary" it measured was that distractor, see docs/amd-hip-port.md 17).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=${1:?kvmem or stock}
BIN=$(bash "$ROOT/tests-data/lmstudio-pack.sh") || exit 1
MODEL="${KVMEM_MOE_GGUF:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf}"
[ -f "$ROOT/tests-data/ladder_30000.json" ] || python "$ROOT/tests-data/gen_ladder.py"

cd "$BIN" || exit 1
LOG="$ROOT/tests-data/ladder-$MODE.log"
taskkill //IM llama-server.exe //F >/dev/null 2>&1
sleep 4
ENV_KVMEM=1
[ "$MODE" = stock ] && ENV_KVMEM=0
LLAMA_ARG_N_CPU_MOE=99 KVMEM_ENABLE=$ENV_KVMEM ./llama-server.exe \
  -m "$(cygpath -m "$MODEL" 2>/dev/null || echo "$MODEL")" \
  --jinja -c 262144 -ngl 99 -b 2048 -ub 512 --flash-attn on --kv-unified --kv-offload \
  --ctx-checkpoints 32 --load-mode mmap+mlock --n-cpu-moe 0 --parallel 1 --reasoning off \
  --port 18499 --no-webui > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 40); do sleep 4; grep -qa listening "$LOG" && break; done
grep -qa listening "$LOG" || { echo "SERVER FAILED"; tail -6 "$LOG" | tr -d '\000'; kill $SRV 2>/dev/null; exit 1; }
echo "[$MODE] KVMem tier lines: $(tr -d '\000' < "$LOG" | grep -ac KVMEM_TIERS)"
for d in ${DEPTHS:-30000 60000 120000 240000}; do
  curl -s -m 1500 -o "$ROOT/tests-data/ladder_${MODE}_$d.json" \
    http://127.0.0.1:18499/v1/chat/completions -H "Content-Type: application/json" \
    --data "@$ROOT/tests-data/ladder_$d.json"
  python - "$ROOT/tests-data/ladder_${MODE}_$d.json" "$d" "$MODE" <<'PY'
import json, sys
serial = {'30000': 'ladder-alpha-100300', '60000': 'ladder-bravo-200600',
          '120000': 'ladder-carlo-301200', '240000': 'ladder-delta-402400'}[sys.argv[2]]
try:
    r = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception as e:
    print(f'{sys.argv[3]} depth={sys.argv[2]}: no result ({e})'); raise SystemExit
if 'choices' not in r:
    print(f'{sys.argv[3]} depth={sys.argv[2]}: ERROR {str(r)[:120]}'); raise SystemExit
t = r['choices'][0]['message'].get('content') or ''
print(f"{sys.argv[3]} depth={sys.argv[2]} hit={serial in t} tok={r['usage']['prompt_tokens']}")
PY
done
kill $SRV 2>/dev/null
echo "[$MODE] DONE"
