#!/usr/bin/env python
"""Decoy-free depth ladder payloads: one unique serial per depth, filler
sanitised so no other word-word-digits pattern exists in the prompt."""
import json
import pathlib
import re

FILLER_SRC = pathlib.Path("tests-data/bench_128k.txt")
if not FILLER_SRC.exists():
    raise SystemExit("missing tests-data/bench_128k.txt - run: python tests-data/gen_bench.py")
BYTES_PER_TOKEN = 7.2

# bench_128k.txt carries needles planted by earlier runs, e.g.
#   "The secret passphrase for benchmark run CHARLIE is: emerald-falcon-802."
# Repeating it 6x over a 240K prompt gives a distractor six votes to the target's
# one, so a recall test measures the distractors. Drop any line that looks like a
# planted-value sentence, then assert only the new needle matches.
BAD_LINE = re.compile(r"(is: [a-z]+-[a-z]+-\d{3,6})|(secret passphrase)|(archive serial)", re.I)
FILLER = "".join(
    line + "\n" for line in FILLER_SRC.read_text(encoding="utf-8", errors="replace").splitlines()
    if not BAD_LINE.search(line))
for tokens, serial in ((30000, "ladder-alpha-100300"), (60000, "ladder-bravo-200600"),
                       (120000, "ladder-carlo-301200"), (240000, "ladder-delta-402400")):
    target = int(tokens * BYTES_PER_TOKEN)
    reps = target // len(FILLER.encode("utf-8")) + 1
    filler = ("\n\n".join([FILLER] * reps))[:target]
    cut = filler.rfind("\n\n", 0, int(len(filler) * 0.5))
    needle = f"The archive serial for the Kepler drift station is: {serial}. Remember it exactly."
    prompt = (filler[:cut] + "\n\n" + needle + "\n\n" + filler[cut + 2:]
              + "\n\nQuestion: What is the archive serial for the Kepler drift station stated"
                " in this document? Answer with the serial only.")
    serials = set(re.findall(r"[a-z]+-[a-z]+-\d{3,6}", prompt))
    frames = prompt.count("station is:")
    assert serials == {serial} and frames == 1, (serials, frames)
    out = pathlib.Path(f"tests-data/ladder_{tokens}.json")
    out.write_text(json.dumps({"messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 64, "temperature": 0.0}, ensure_ascii=False),
                   encoding="utf-8")
    print(f"{out.name}: {len(prompt)} chars, serial {serial}, needle at {cut / len(filler):.0%}")
