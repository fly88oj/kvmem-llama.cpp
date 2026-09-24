"""Generate a long-context needle-recall prompt for the KVMem HIP smoke test.

Needle planted in block ~0; filler ~3k tokens; question at the end. With
--kvmem-budget 512 the needle block MUST be evicted to the host arena and
retrieved by query-score only — correct answer == retrieval pipeline works.
"""
import random

random.seed(7)
needle = "The launch code for the Mars probe Zephyr-9 is: amber-jaguar-4417."
topics = ["glacier mapping", "deep-sea vents", "medieval bridges", "quantum error correction",
          "bird migration", "fermentation chemistry", "radio telescopes", "urban drainage",
          "volcanic glass", "bee navigation", "ice core dating", "suspension bridges"]
paras = []
for i in range(34):
    t = topics[i % len(topics)]
    words = [random.choice(["the", "and", "of", "in", "for", "with", "data", "study",
                            "shows", "system", "process", "observed", "measured", "model",
                            "because", "however", "therefore", "research", "analysis"]) for _ in range(55)]
    paras.append(f"On {t}, note {i}: " + " ".join(words) + ".")
body = "\n\n".join(paras)
# Put the needle after ~5 paragraphs so it falls outside the always-kept sink
# block and the recent suffix — only query-based retrieval can bring it back.
mid = "\n\n".join(paras[:5])
tail = "\n\n".join(paras[5:])
prompt = (
    f"{mid}\n\n{needle}\n\nRemember it exactly; it will be asked later.\n\n"
    f"{tail}\n\n"
    f"Question: What is the launch code for the Mars probe Zephyr-9 stated in the middle of this document? Answer with the code only."
)
with open("needle_prompt.txt", "w", encoding="utf-8") as fh:
    fh.write(prompt)
print("chars:", len(prompt), "approx tokens:", len(prompt) // 4)
