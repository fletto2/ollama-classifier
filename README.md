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

`input` can also be a list of strings, which returns one result per string. A head answers a yes/no (`noul`), multiple-choice (`choice`) or ordinal (`score`) question, in the answer shapes of the llama.cpp-classifier README. Heads are trained with `llama-classifier train` from llama.cpp-classifier.

**Classifier endpoints without a head, features and live head training:** every classifier endpoint of llama.cpp-classifier's llama-server is reachable through Ollama. The request body is passed through as is (with `model` and `keep_alive` handled by Ollama), and each needs its server setting:

| route | llama-server endpoint | setting |
|---|---|---|
| `POST /api/classify` with `"questions"` | `/classify`, questions answered by similarity to their answers' descriptions or example texts, no head needed | `OLLAMA_CLASSIFIER_FEATURES=1` |
| `POST /api/features` | `/features`, pooled hidden states of chosen layers | `OLLAMA_CLASSIFIER_FEATURES=1` |
| `POST /api/options` | `/options`, log-probabilities of fixed answers after a prompt | `OLLAMA_OPTIONS=1` |
| `POST /api/classify/train` | `/classify/train`, fit a head and register it live (`"save": true` writes it to `<models>/classifier-train`) | `OLLAMA_CLASSIFIER_TRAIN=1` |

`OLLAMA_CLASSIFIER_CTX` and `OLLAMA_CLASSIFIER_UBATCH` set the classifier contexts' maximum input length and compute step (llama-server `--classifier-ctx`, `--classifier-ubatch`). Each enabled endpoint adds a context to every runner, so enable only what is used.

**LoRA training on a loaded model:** start the server with `OLLAMA_LORA_TRAIN=1`, then run

```shell
ollama train BASE NEW -f data.txt [--rank 8 --lr 1e-4 --epochs 1 --num-ctx 256 --priority idle]
```

or call `POST /api/train`. This trains an adapter on the running base model and saves `NEW` = `BASE` + the adapter.
- With `--priority idle`, training steps run only while the model is not serving requests.
- `OLLAMA_LORA_TRAIN=1` applies to every runner the server starts. They load weights without CPU repacking, which slows prompt processing on CPU for all models. The training context also needs memory beyond Ollama's estimate.
- `BASE` must be a GGUF model without adapters, and `NEW` must be a different name; an existing model called `NEW` is replaced.
- Recurrent/hybrid (e.g. Qwen3.5), MoE and diffusion models can't be trained: some of their ops have no backward pass.

**Speculative decoding without a draft model:** `OLLAMA_SPEC_TYPE=ngram-cache` (or `ngram-mod`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`) makes runners draft from n-grams already in the prompt and output, for models that have no MTP head or draft model. The `draft_num_predict` option sets the draft length (default 4). Output is unchanged under greedy sampling. It pays on output that copies the input (rewritten code or listings); on free text it is about neutral.

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
  - `OLLAMA_LORA_TRAIN`, `OLLAMA_SPEC_TYPE`, `OLLAMA_CLASSIFIER_FEATURES`, `OLLAMA_CLASSIFIER_TRAIN`, `OLLAMA_OPTIONS`, `OLLAMA_CLASSIFIER_CTX`, `OLLAMA_CLASSIFIER_UBATCH`
  - `POST /api/features`, `POST /api/options`, `POST /api/classify/train`

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
