# ollama-classifier

This is [Ollama](https://github.com/ollama/ollama), built on [llama.cpp-classifier](https://github.com/fletto2/llama.cpp-classifier) instead of upstream llama.cpp. llama.cpp-classifier is upstream llama.cpp plus **per-context early exit**: `llama_set_n_layer_exit(ctx, L)` lets a classifier context share one loaded model with text generation. See that repo's README for how the classifier works. Everything else here is unchanged upstream Ollama; for installation, usage, the API and the full documentation, see the [official Ollama README](https://github.com/ollama/ollama/blob/main/README.md).

## What differs from upstream Ollama

- `LLAMA_CPP_VERSION`: pins a llama.cpp-classifier commit (the tip of its `master`, because the clone is shallow).
- `llama/server/CMakeLists.txt`, `cmake/local.cmake`: fetch llama.cpp from `fletto2/llama.cpp-classifier`.
- `llama/compat/001-llama-cpp-hooks.patch`, `002-clef.patch`: Ollama's patches, rebased onto that base. Upstream llama.cpp now has its own Clef architecture. Ollama-converted Clef models (`qwen35` backbone + `clef.*` tensors) still use Ollama's head through `score_fields`, with the same scores as upstream Ollama.

## Build

The same as upstream Ollama ([docs/development.md](docs/development.md)):

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To check that only the patched llama.cpp source is right (fetched at the pinned commit, both compat patches applied):

```shell
cmake -S llama/server --preset cpu
git -C build/llama-server-cpu/_deps/llama_cpp-src log -1
```
