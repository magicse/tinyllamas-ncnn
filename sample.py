"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import ModelArgs, Transformer
from torch.nn import functional as F
from tokenizer import Tokenizer

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
start = "One day," # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = ModelArgs(**checkpoint['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
enc = Tokenizer()

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = enc.encode(start, bos=True, eos=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)

def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    Also note this is a super inefficient version of sampling with no key/value cache.
    """
    idx_cond_new = None
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(0) <= model.params.max_seq_len else idx[:, -model.params.max_seq_len:]
        idx_cond_new = torch.cat((torch.full((model.params.max_seq_len - idx.size(0),), 0, dtype=torch.long), idx_cond), dim=0)
        idx_cond = idx_cond_new
        # forward the model to get the logits for the index in the sequence
        logits = model(idx_cond)
        logits = logits[-1, :]
        if temperature == 0.0:
            # "sample" the single most likely index
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[[-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=0)

    mod = torch.jit.trace(model, idx_cond_new)
    mod.save("model.pt")

    return idx

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y.tolist()))
            print('---------------')
