"""gguf_meta.py - read GGUF metadata from the header ONLY (no model load).

Zero-risk: reads just the header/metadata KV section, never mmaps tensors or
allocates model memory. Used to estimate KV-cache size before committing to a
large-context run.

Usage: python gguf_meta.py <model.gguf>
"""
import struct
import sys


def main(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit("not a GGUF file: " + path)
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 0: return struct.unpack("<B", f.read(1))[0]
            if t == 1: return struct.unpack("<b", f.read(1))[0]
            if t == 2: return struct.unpack("<H", f.read(2))[0]
            if t == 3: return struct.unpack("<h", f.read(2))[0]
            if t == 4: return struct.unpack("<I", f.read(4))[0]
            if t == 5: return struct.unpack("<i", f.read(4))[0]
            if t == 6: return struct.unpack("<f", f.read(4))[0]
            if t == 7: return struct.unpack("<?", f.read(1))[0]
            if t == 8: return rstr()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                return [rval(et) for _ in range(n)]
            if t == 10: return struct.unpack("<Q", f.read(8))[0]
            if t == 11: return struct.unpack("<q", f.read(8))[0]
            if t == 12: return struct.unpack("<d", f.read(8))[0]
            raise ValueError("unknown gguf value type %d" % t)

        print("GGUF v%d  tensors=%d  metadata_kv=%d" % (version, tensor_count, kv_count))
        wanted = ("architecture", "block_count", "context_length", "head_count",
                  "key_length", "value_length", "feed_forward_length", "expert",
                  "layer_types", "embedding_length", "max_position", "recurrent",
                  "linear_attention", "attention.", "mtp", "nextn", "time_decay",
                  "conv_kernel", "state_size", "full_attention", "sliding")
        for _ in range(kv_count):
            k = rstr()
            t = struct.unpack("<I", f.read(4))[0]
            v = rval(t)
            if not any(w in k for w in wanted):
                continue
            if isinstance(v, list):
                # Summarize long arrays (e.g. per-layer layer_types).
                if len(v) > 12:
                    uniq = {}
                    for x in v:
                        uniq[x] = uniq.get(x, 0) + 1
                    print("  %s = [%d items] counts=%s head=%s" % (k, len(v), uniq, v[:8]))
                else:
                    print("  %s = %s" % (k, v))
            else:
                print("  %s = %s" % (k, v))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python gguf_meta.py <model.gguf>")
    main(sys.argv[1])
