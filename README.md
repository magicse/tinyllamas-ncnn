# tinyllamas-ncnn

This is a repository hosting code for converting tinyllamas models into ncnn format and inference code using ncnn.

Changes to the model:

- Removed batching to avoid tensors of rank 5 and up
- Moved sampling into sample.py
- Reverted to using manual implementation of flash attention
- Always pad input to transformer to the full length of context length to avoid variable shape inputs
- Applied workaround as described in <https://github.com/Tencent/ncnn/issues/4937>

## Usage

### Export TorchScript

First put desired model described in <https://github.com/karpathy/llama2.c#models> into /out/ckpt.pt. Then create a Python venv, enter it, and install Python dependencies:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run sample.py and the resulting TorchScript will be in model.pt.

### Convert model into ncnn format

```
pnnx model.pt inputshape=[xxx]i32
```

You need to replace `xxx` with the context length of your chosen model.

The resulting model.ncnn.bin and model.ncnn.param is the model in ncnn format.

### Compile the inference binary

You can either compile it using CMake with ncnn as a dependency or link the library yourself. Before compiling, adjust ctx_length in tinyllamas.cpp to the context length of your model.

#### Compile with CMake

This follows the standard procedure for compiling CMake projects.

#### Link ncnn manually

```
c++ tinyllamas.cpp ~/ncnn/build/src/libncnn.a -I ~/ncnn/src -I ~/ncnn/build/src/ -o tinyllamas -fopenmp
```

### Use the resulting binary

When run with a wrong number of arguments, the binary prints out usage information.

## License

MIT
