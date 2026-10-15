"""Generate filler prompts of ~8K / ~32K / ~128K tokens for the KVMem A/B benchmark.

Each document ends with a needle question; a needle is planted mid-document so
every run also sanity-checks retrieval. Token counts are calibrated from the
engine's own prompt_n logs afterwards.
"""
import random

random.seed(11)
WORDS = ["system", "analysis", "measure", "observe", "process", "research", "model",
         "data", "theory", "experiment", "result", "method", "approach", "evidence",
         "conclude", "derive", "estimate", "calibrate", "protocol", "document",
         "section", "chapter", "figure", "table", "equation", "hypothesis", "sample"]

def filler_paragraph(i: int) -> str:
    n = random.randint(48, 72)
    return f"Section {i}. " + " ".join(random.choices(WORDS, k=n)) + "."

def doc(target_words: int, needle: str) -> str:
    paras = []
    words = 0
    i = 0
    mid_done = False
    while words < target_words:
        p = filler_paragraph(i)
        paras.append(p)
        words += p.count(" ") + 1
        if not mid_done and words >= target_words // 2:
            paras.append(needle)
            mid_done = True
        i += 1
    return "\n\n".join(paras)

SPECS = {
    "bench_8k.txt":   (2200,  "The secret passphrase for benchmark run ALPHA is: turquoise-cobra-771."),
    "bench_32k.txt":  (8800,  "The secret passphrase for benchmark run BRAVO is: sapphire-tiger-349."),
    "bench_128k.txt": (36000, "The secret passphrase for benchmark run CHARLIE is: emerald-falcon-802."),
}
Q = "\n\nQuestion: What is the secret passphrase stated in the middle of this document? Answer with the passphrase only."
for name, (tw, needle) in SPECS.items():
    text = doc(tw, needle) + Q
    with open(name, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(name, "chars:", len(text), "words:", text.count(" ") + 1)
