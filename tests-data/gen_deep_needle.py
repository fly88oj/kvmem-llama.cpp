"""Generate a deep-fill needle prompt of a target token size.

Repeats tests-data/bench_128k.txt (measured 40957 tokens / 0.28MB) to reach the
requested token budget, plants a unique needle at a given depth fraction, and
appends the recall question. Used for 256K/512K/1M-context KVMem tier tests.
"""
import argparse
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / "bench_128k.txt"
if not BASE.exists():
    raise SystemExit(f"missing {BASE} - run: python tests-data/gen_bench.py")
BYTES_PER_TOKEN = 7.2  # measured for bench_128k.txt (293KB / 40957 tokens)

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, required=True, help="target prompt tokens (approx)")
p.add_argument("--needle-at", type=float, default=0.5, help="depth fraction 0..1 for the needle")
p.add_argument("--needle", default="The archive serial for the Kepler drift station is: cobalt-heron-9136.")
p.add_argument("--out", required=True)
a = p.parse_args()
m = re.search(r"[a-z]+-[a-z]+-\d{3,6}", a.needle)
a.needle_serial = m.group(0) if m else ""

body = BASE.read_text(encoding="utf-8", errors="replace")
# The stock filler carries a needle planted by an earlier run
# ("The secret passphrase for benchmark run CHARLIE is: emerald-falcon-802."). Repeating
# it across a 1M prompt gives that distractor thousands of votes against the target's
# one, so a recall test measures the distractors. Drop the whole line, not just the
# serial: replacing only the serial leaves six fake passphrase sentences behind.
bad = re.compile(r"(is: [a-z]+-[a-z]+-\d{3,6})|(secret passphrase)|(archive serial)", re.I)
body = "".join(line + "\n" for line in body.splitlines() if not bad.search(line))
target_bytes = int(a.tokens * BYTES_PER_TOKEN)
reps = target_bytes // len(body.encode("utf-8")) + 1
filler = ("\n\n".join([body] * reps))[:target_bytes]

cut = int(len(filler) * a.needle_at)
# snap the cut to a paragraph boundary so the needle sits cleanly between blocks
cut = filler.rfind("\n\n", 0, cut)
if cut < 0:
    cut = len(filler) // 2

prompt = (
    filler[:cut]
    + "\n\n" + a.needle + " Remember it exactly; it will be asked later.\n\n"
    + filler[cut:]
    + "\n\nQuestion: What is the archive serial for the Kepler drift station stated in this document? Answer with the serial only."
)
pathlib.Path(a.out).write_text(prompt, encoding="utf-8")
found = set(re.findall(r"[a-z]+-[a-z]+-\d{3,6}", prompt))
assert found == {a.needle_serial}, f"prompt carries {found}, expected one serial"
print(f"wrote {a.out}: {len(prompt)/1024:.0f}KB ~{len(prompt)/BYTES_PER_TOKEN:.0f} tokens, needle at {a.needle_at:.0%}")
