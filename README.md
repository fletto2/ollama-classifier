# ollama-classifier

This is [Ollama](https://github.com/ollama/ollama), built on [llama.cpp-classifier](https://github.com/fletto2/llama.cpp-classifier) instead of upstream llama.cpp. llama.cpp-classifier adds **classifier heads** that share the loaded model with text generation (per-context early exit) and **embedded GGUF LoRA training** in C/C++ on a frozen, even quantized, base. See that repo's README for both.

Everything else is unchanged upstream Ollama. For installation, usage, the API and the full documentation, see the [official Ollama README](https://github.com/ollama/ollama/blob/main/README.md).

## Additions

**LoRA adapters:** `ADAPTER adapter.gguf` in a Modelfile adds a GGUF LoRA adapter to a GGUF model. Such adapters come e.g. from `llama-finetune --lora-rank` or `ollama train`.

**Classifier heads:** `CLASSIFIER head.gguf` adds a GGUF classifier head (`general.type = classifier`); a model can have several. The heads run on the loaded model, next to generation:

```shell
curl localhost:11434/api/classify -d '{"model": "my-model", "input": "text"}'
# {"model":"my-model","answers":{"relevant":{"type":"noul","noul":0.97}},"usage":{"input_tokens":3,"output_tokens":0}}
```

`input` can also be a list of strings, which returns one result per string.

**LoRA training on a loaded model:** start the server with `OLLAMA_LORA_TRAIN=1`, then run

```shell
ollama train BASE NEW -f data.txt [--rank 8 --lr 1e-4 --epochs 1 --num-ctx 256 --priority idle]
```

or call `POST /api/train`. This trains an adapter on the running base model and saves `NEW` = `BASE` + the adapter.
- With `--priority idle`, training steps run only while the model is not serving requests.
- `OLLAMA_LORA_TRAIN=1` applies to every runner the server starts. They load weights without CPU repacking, which slows prompt processing on CPU for all models. The training context also needs memory beyond Ollama's estimate.
- `BASE` must be a GGUF model without adapters, and `NEW` must be a different name; an existing model called `NEW` is replaced.
- Recurrent/hybrid (e.g. Qwen3.5), MoE and diffusion models can't be trained: some of their ops have no backward pass.

## What differs from upstream Ollama

- **llama.cpp source:**
  - `LLAMA_CPP_VERSION` pins a llama.cpp-classifier commit.
  - `llama/server/CMakeLists.txt` and `cmake/local.cmake` fetch it from `fletto2/llama.cpp-classifier` with a full clone, so the pin may be any commit of the fork.
  - Ollama's two compat patches in `llama/compat/` (the hooks patch and the decision-head patch) are rebased onto that base.
  - Ollama-converted decision models (a `qwen35` backbone with the decision head stored as separate tensors) still use Ollama's head through `score_fields`, with the same scores as upstream Ollama.
- **Go:**
  - `ADAPTER` / `CLASSIFIER` in the Modelfile parser, `ollama create` and the model layers (`application/vnd.ollama.image.adapter`, `application/vnd.ollama.image.classifier`)
  - the runner flags `--lora`, `--classifier`, `--lora-train`, `--lora-train-dir` (`<models>/lora-train`)
  - `POST /api/classify`, `POST /api/train`, `ollama train`
  - `OLLAMA_LORA_TRAIN`

## Build

The same as upstream Ollama ([docs/development.md](docs/development.md)):

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To check only the patched llama.cpp source (fetched at the pinned commit, both compat patches applied):

```shell
cmake -S llama/server --preset cpu
git -C build/llama-server-cpu/_deps/llama_cpp-src log -1
```
