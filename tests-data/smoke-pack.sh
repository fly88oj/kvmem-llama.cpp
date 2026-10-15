#!/usr/bin/env bash
# Pack-level acceptance for the installed KVMem LM Studio backend.
#   A) --parallel 4 must state out loud that KVMem is off (and still serve)
#   B) --parallel 1 must activate KVMem and survive a second request on the same slot
# Both cases use the decoy-free ladder payloads at 29K and 58K depth.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN=$(bash "$ROOT/tests-data/lmstudio-pack.sh") || exit 1
MODEL="${KVMEM_MOE_GGUF:-$HOME/.lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF/gemma-4-26B-A4B-it-QAT-Q4_0.gguf}"
[ -f "$ROOT/tests-data/ladder_30000.json" ] || python "$ROOT/tests-data/gen_ladder.py"
cd "$BIN" || exit 1
echo "pack: $BIN"

run_case() {
    local tag=$1 port=$2
    local log="$ROOT/tests-data/smoke-$tag.log"
    taskkill //IM llama-server.exe //F >/dev/null 2>&1
    sleep 4
    LLAMA_ARG_N_CPU_MOE=99 KVMEM_BUDGET=0 ./llama-server.exe \
      -m "$(cygpath -m "$MODEL" 2>/dev/null || echo "$MODEL")" \
      --jinja -c 262144 -ngl 99 -b 2048 -ub 512 --flash-attn on --kv-unified --kv-offload \
      --ctx-checkpoints 32 --load-mode mmap+mlock --n-cpu-moe 0 --reasoning off \
      --parallel "$3" --port "$port" --no-webui > "$log" 2>&1 &
    local srv=$!
    for i in $(seq 1 30); do sleep 4; grep -qa listening "$log" && break; done
    if ! grep -qa listening "$log"; then
        echo "$tag: SERVER FAILED"; tail -4 "$log" | tr -d '\000'; kill $srv 2>/dev/null; return 1
    fi
    echo "$tag: tiers=$(tr -d '\000' < "$log" | grep -ac KVMEM_TIERS) kvmem_off=$(tr -d '\000' < "$log" | grep -ac 'KVMem is disabled')"
    for d in 30000 60000; do
        curl -s -m 600 -o "$ROOT/tests-data/smoke-${tag}-${d}.json" \
            -w "$tag $d http=%{http_code} " "http://127.0.0.1:$port/v1/chat/completions" \
            -H "Content-Type: application/json" --data "@$ROOT/tests-data/ladder_$d.json"
        python - "$ROOT/tests-data/smoke-${tag}-${d}.json" "$d" <<'PY'
import json, sys
serial = {'30000': 'ladder-alpha-100300', '60000': 'ladder-bravo-200600'}[sys.argv[2]]
r = json.load(open(sys.argv[1], encoding='utf-8'))
if 'choices' not in r:
    print(f'resp-error={str(r.get("error"))[:70]} ', end=''); raise SystemExit
m = r['choices'][0]['message']
t = (m.get('content') or '') + (m.get('reasoning_content') or '')
print(f'hit={serial in t} ', end='')
PY
    done
    echo ""
    tr -d '\000' < "$log" | grep -aE "KVMem is disabled|has no GPU slot|Context size" | head -2
    kill $srv 2>/dev/null
    sleep 3
}

run_case parallel4 18511 4
run_case parallel1 18512 1
echo SMOKE-DONE
