import struct
import torch
import numpy as np
from pathlib import Path
from functools import reduce

path = Path("llama-2-7b.bin")
out = path.with_suffix(".pt")

f = open(path, "rb")

dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, ctx_len = struct.unpack('iiiiiii', f.read(4*7))
shared = vocab_size > 0
vocab_size = abs(vocab_size)

print(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, ctx_len)

# embedding
embedding = np.frombuffer(f.read(4 * vocab_size * dim), dtype=np.float32).reshape(vocab_size, dim)
embedding = torch.tensor(embedding)

def read_tensor(f, shape):
    size = reduce(lambda x, y: x * y, shape, 1)
    return torch.from_numpy(np.frombuffer(f.read(4 * size), dtype=np.float32).reshape(shape))

# rms_att_weight
rms_att_weight = [None for _ in range(n_layers)]
for i in range(n_layers):
    rms_att_weight[i] = read_tensor(f, (dim,))

# wq
wq = [None for _ in range(n_layers)]
for i in range(n_layers):
    wq[i] = read_tensor(f, (dim, dim))

# wk
wk = [None for _ in range(n_layers)]
for i in range(n_layers):
    wk[i] = read_tensor(f, (dim, dim))

# wv
wv = [None for _ in range(n_layers)]
for i in range(n_layers):
    wv[i] = read_tensor(f, (dim, dim))

# wo
wo = [None for _ in range(n_layers)]
for i in range(n_layers):
    wo[i] = read_tensor(f, (dim, dim))

# rms_ffn_weight
rms_ffn_weight = [None for _ in range(n_layers)]
for i in range(n_layers):
    rms_ffn_weight[i] = read_tensor(f, (dim,))

# w1
w1 = [None for _ in range(n_layers)]
for i in range(n_layers):
    w1[i] = read_tensor(f, (hidden_dim, dim))

print("w1")

# w2
w2 = [None for _ in range(n_layers)]
for i in range(n_layers):
    w2[i] = read_tensor(f, (dim, hidden_dim))

# w3
w3 = [None for _ in range(n_layers)]
for i in range(n_layers):
    w3[i] = read_tensor(f, (hidden_dim, dim))

# rms_final_weight
rms_final_weight = read_tensor(f, (dim,))

head_size = dim // n_heads
# freq_trans_real
freq_trans_real = read_tensor(f, (ctx_len, head_size // 2))

# freq_trans_imag
freq_trans_imag = read_tensor(f, (ctx_len, head_size // 2))

# unembedding
unembedding = read_tensor(f, (vocab_size, dim))

print("read")

f.close()

# write .pt
state_dict = {}
state_dict["tok_embeddings.weight"] = embedding
for i in range(n_layers):
    state_dict[f"layers.{i}.attention.wq.weight"] = wq[i]
    state_dict[f"layers.{i}.attention.wk.weight"] = wk[i]
    state_dict[f"layers.{i}.attention.wv.weight"] = wv[i]
    state_dict[f"layers.{i}.attention.wo.weight"] = wo[i]
    state_dict[f"layers.{i}.feed_forward.w1.weight"] = w1[i]
    state_dict[f"layers.{i}.feed_forward.w2.weight"] = w2[i]
    state_dict[f"layers.{i}.feed_forward.w3.weight"] = w3[i]
    state_dict[f"layers.{i}.attention_norm.weight"] = rms_att_weight[i]
    state_dict[f"layers.{i}.ffn_norm.weight"] = rms_ffn_weight[i]
state_dict["norm.weight"] = rms_final_weight
state_dict["output.weight"] = unembedding

checkpoint = {
    "model_args": {
        "dim": dim, 
        "n_layers": n_layers, 
        "n_heads" : n_heads, 
        "n_kv_heads": n_kv_heads, 
        "vocab_size": vocab_size, 
        "multiple_of": 256,
        "max_seq_len": ctx_len,
        "dropout": 0,
    },
    "model": state_dict,
}

print("save")
torch.save(checkpoint, out)
