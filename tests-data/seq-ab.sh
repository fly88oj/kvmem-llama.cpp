#!/usr/bin/env bash
# Two-request sequence on one slot - the shape that broke multi-turn chats before
# the adapter mirrored suffix cache trims (docs/amd-hip-port.md 17.2).
#   usage: bash tests-data/seq-ab.sh kvmem|stock [extra llama-server args...]
# Runs against the installed KVMem pack; set KVMEM_PACK to point elsewhere.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=${1:?kvmem or stock}
shift
BIN=$(bash "$ROOT/tests-data/lmstudio-pack.sh") || exit 1
MODEL="${KVMEM_MOE_GGUF:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf}"
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }
[ -f "$ROOT/tests-data/ladder_30000.json" ] || python "$ROOT/tests-data/gen_ladder.py"

cd "$BIN" || exit 1
taskkill //IM llama-server.exe //F >/dev/null 2>&1
sleep 4
LOG="$ROOT/tests-data/seq-$MODE.log"
ENV_KVMEM=1
[ "$MODE" = stock ] && ENV_KVMEM=0
LLAMA_ARG_N_CPU_MOE=99 KVMEM_ENABLE=$ENV_KVMEM ./llama-server.exe \
  -m "$(cygpath -m "$MODEL" 2>/dev/null || echo "$MODEL")" \
  --jinja -c 262144 -ngl 99 -b 2048 -ub 512 --flash-attn on --kv-unified --kv-offload \
  --ctx-checkpoints 32 --load-mode mmap+mlock --n-cpu-moe 0 --parallel 1 --reasoning off \
  --port 18500 --no-webui "$@" > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 40); do sleep 4; grep -qa listening "$LOG" && break; done
grep -qa listening "$LOG" || { echo "SERVER FAILED"; tail -8 "$LOG" | tr -d '\000'; kill $SRV 2>/dev/null; exit 1; }
echo "[$MODE] kvmem tier lines: $(tr -d '\000' < "$LOG" | grep -ac KVMEM_TIERS)"
for d in 30000 60000; do
  curl -s -m 900 -o "$ROOT/tests-data/seq-$MODE-$d.json" \
    -w "[$MODE] $d http=%{http_code} " http://127.0.0.1:18500/v1/chat/completions \
    -H "Content-Type: application/json" --data "@$ROOT/tests-data/ladder_$d.json"
  python - "$ROOT/tests-data/seq-$MODE-$d.json" "$d" <<'PY'
import json, sys
serial = {'30000': 'ladder-alpha-100300', '60000': 'ladder-bravo-200600'}[sys.argv[2]]
r = json.load(open(sys.argv[1], encoding='utf-8'))
if 'choices' not in r:
    print(f"FAIL {str(r.get('error', r))[:100]}")
else:
    m = r['choices'][0]['message']
    t = (m.get('content') or '') + (m.get('reasoning_content') or '')
    print(f"hit={serial in t} tok={r['usage']['prompt_tokens']} "
          f"cached={r['usage']['prompt_tokens_details'].get('cached_tokens')}")
PY
done
tr -d '\000' < "$LOG" | grep -aE "has no GPU slot|Context size" | tail -3
kill $SRV 2>/dev/null
echo "[$MODE] DONE"
