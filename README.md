# finetune-fun 🏴‍☠️

A minimal, runnable **QLoRA finetuning** starter for small **Qwen** models, built on the
Hugging Face stack (`transformers` + `peft` + `trl` + `datasets`) with **Unsloth** for
speed and **W&B / TensorBoard** for tracking progress.

The default target is [`ProCreations/grug-think`](https://huggingface.co/datasets/ProCreations/grug-think)
— a **tool-use / function-calling** dataset where the assistant reasons in caveman-style
`<think>` traces before emitting structured tool calls. A tiny local pirate-persona
dataset (`data/example_dataset.jsonl`) is also included as a simpler fallback.

---

## Why rent a GPU (and not use the Mac)

QLoRA relies on `bitsandbytes` 4-bit quantization and Unsloth, which are **CUDA-only** —
they don't run on Apple Silicon. Rent a small NVIDIA GPU instead. It's cheap:

| Provider | Good GPU pick | ~Price | Notes |
|----------|---------------|--------|-------|
| [RunPod](https://runpod.io) | RTX 4090 (24GB) | ~$0.35–0.45/hr | Easiest; pick a PyTorch template |
| [Vast.ai](https://vast.ai) | RTX 4090 / 3090 | ~$0.20–0.40/hr | Cheapest; slightly more manual |
| [Lambda](https://lambdalabs.com) | A10 / A100 | ~$0.75/hr+ | Clean UX |

A full run of this example takes **a few minutes**, so a `$50` budget is dozens of hours
of experimentation. **Stop/terminate the instance when done** so you're not billed idle.

> Edit `data/` on your Mac, push to GitHub, then `git clone` on the GPU box — or just
> upload the files via the provider's Jupyter/web terminal.

---

## Setup (on the rented GPU)

```bash
git clone <your-repo-url> && cd finetune-fun
pip install -r requirements.txt
```

If the plain `unsloth` install gives trouble, use the CUDA-matched install (check the box's
CUDA/torch with `nvidia-smi` and `python -c "import torch; print(torch.__version__)"`), e.g.:

```bash
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

## Train

The default dataset is `ProCreations/grug-think` (pulled straight from the Hub), subsampled
to 3,000 rows — the full 101k rows is far more than a first run needs.

```bash
# 1. SANITY CHECK FIRST (no GPU work): render one example and eyeball it.
#    Tool-calling chat templates are finicky — confirm the <think> tags, tool
#    definitions, and tool calls render before spending GPU time.
python train.py --dataset ProCreations/grug-think --preview

# 2. Local dashboard (no signup): logs to ./outputs, view with `tensorboard --logdir outputs`
python train.py --report-to tensorboard

# 3. Hosted dashboard: run `wandb login` once, then
python train.py --report-to wandb
```

Useful flags: `--subsample 5000` (or `0` for all 101k), `--model unsloth/Qwen3-4B`,
`--epochs 2`, `--lora-rank 32`, `--max-seq-len 4096` (keeps longer conversations, needs
more VRAM). Run `python train.py --help` for the full list.

> **Why subsample?** `grug-think` teaches a *skill* (agentic tool use), not just a style,
> so you want thousands of examples — but not all 101k for a first pass. 3–5k rows trains
> in well under an hour on a 4090 and is enough to see the model pick up the grug `<think>`
> reasoning + tool-call format. Scale up later if results look promising.

The trained **LoRA adapter** lands in `outputs/lora_adapter/` (only a few MB — it's just
the small trained matrices, not the whole model).

## Chat with it

```bash
python inference.py --adapter outputs/lora_adapter
```

## Measure whether it worked

Plain chat won't test what grug-think teaches. `eval.py` feeds held-out prompts *with* tool
definitions and checks that the model produces (1) a `<think>` block, (2) a valid
`<tool_call>` with parseable JSON, and (3) the right tool name:

```bash
python eval.py --adapter outputs/lora_adapter

# Compare against the un-finetuned base model on the same prompts:
python eval.py --adapter outputs/lora_adapter --compare-base unsloth/Qwen2.5-3B-Instruct
```

The `--compare-base` run is the honest test: the finetune should hit those three checks more
consistently than the base model.

---

## Dataset formats

`train.py` accepts either a **Hub dataset id** or a **local `.jsonl` path**, and both use
the HF conversational format (the chat template is applied for you — never write special
tokens by hand).

**Simple (persona/style)** — one JSON object per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Tool-use (what `grug-think` uses)** — rows also carry a `tools` list, and assistant turns
can include `tool_calls`, with `tool` turns for the results. `train.py` handles all of this
(it passes `tools=` to the chat template and normalizes tool-call arguments):

```json
{"tools": [{"type": "function", "function": {"name": "...", "parameters": {...}}}],
 "messages": [{"role": "assistant", "content": "<think>...</think>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]},
              {"role": "tool", "tool_call_id": "call_1", "content": "{...}"}]}
```

Tips for a first project:
- Data **amount depends on the goal**: ~50–300 for pure style, but **thousands** for a skill
  like tool use. Quality/consistency beats raw volume either way.
- Start with a **small base** (`Qwen2.5-1.5B` / `3B`) for fast iteration, scale up later.
- Always keep a **held-out set** and compare base vs finetuned on the same prompts.

---

## Taking it home (optional): run on your Mac via Ollama

After training, export to GGUF so you can run it locally on Apple Silicon with no GPU:

```python
# on the GPU box, after training — merges LoRA + base and quantizes to GGUF
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("outputs/lora_adapter", load_in_4bit=True)
model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q4_k_m")
```

Download the resulting `.gguf`, then on your Mac: `ollama create bytebeard -f Modelfile`
(a one-line `Modelfile` pointing at the gguf) and `ollama run bytebeard`.

---

## What each file does

| File | Purpose |
|------|---------|
| `train.py` | Loads Qwen in 4-bit, attaches LoRA, renders messages+tools, trains on assistant turns, saves adapter |
| `inference.py` | Interactive chat with the finetuned adapter |
| `eval.py` | Tool-use eval: checks `<think>` + valid tool calls on held-out prompts (base vs finetuned) |
| `data/example_dataset.jsonl` | 15-example pirate-persona demo (fallback / simpler format reference) |
| `requirements.txt` | The HF stack + Unsloth + tracking libraries |
