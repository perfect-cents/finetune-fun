"""
QLoRA finetuning of a small Qwen model with Unsloth + TRL's SFTTrainer.

Run this on a rented NVIDIA GPU (see README). Examples:

    # Function-calling dataset from the Hugging Face Hub (subsampled):
    python train.py --dataset ProCreations/grug-think --subsample 3000 --report-to wandb

    # Preview how one row renders (no GPU work) before committing to a run:
    python train.py --dataset ProCreations/grug-think --preview

    # A local .jsonl file also works:
    python train.py --dataset data/example_dataset.jsonl

Datasets use the Hugging Face "conversational" format: each row has a `messages`
list, and optionally a `tools` list of OpenAI-style function definitions. The
model's own chat template is applied automatically (including tool rendering),
so you never hand-write Qwen's special tokens.
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA finetune a Qwen model with Unsloth.")
    # Base model. Unsloth ships pre-quantized 4-bit repos that download faster.
    # Qwen2.5-Instruct models support tool-calling in their chat template, which
    # this dataset needs. Other choices: unsloth/Qwen2.5-1.5B-Instruct, unsloth/Qwen3-4B
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset", default="ProCreations/grug-think",
                   help="A Hub dataset id (e.g. ProCreations/grug-think) or a local .jsonl path.")
    p.add_argument("--subsample", type=int, default=3000,
                   help="Randomly keep at most this many rows (0 = use all). "
                        "grug-think has 101k rows; a few thousand is plenty for a first run.")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--max-seq-len", type=int, default=2048,
                   help="Rows that tokenize longer than this are dropped, not truncated, "
                        "so tool calls are never cut in half. Raise it (needs more VRAM) "
                        "to keep the longest multi-turn conversations.")
    p.add_argument("--preview", action="store_true",
                   help="Print the first fully-rendered training example and exit (no GPU needed).")

    # LoRA hyperparameters — sensible defaults for a first run.
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)

    # Training schedule.
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="Override epochs if > 0.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)

    # "wandb" for the hosted dashboard, "tensorboard" for a local one, "none" to disable.
    p.add_argument("--report-to", default="tensorboard", choices=["wandb", "tensorboard", "none"])
    p.add_argument("--seed", type=int, default=3407)
    return p.parse_args()


def normalize_messages(messages):
    """Make dataset messages safe for apply_chat_template.

    Chat templates expect each tool call's `arguments` to be a JSON object, but
    datasets often store it as a JSON string. Parse it back to a dict so Qwen's
    template renders it correctly instead of double-encoding it.
    """
    fixed = []
    for m in messages:
        m = dict(m)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            new_calls = []
            for tc in tool_calls:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                arg = fn.get("arguments")
                if isinstance(arg, str):
                    try:
                        fn["arguments"] = json.loads(arg)
                    except (json.JSONDecodeError, TypeError):
                        pass
                tc["function"] = fn
                new_calls.append(tc)
            m["tool_calls"] = new_calls
        fixed.append(m)
    return fixed


def load_hub_dataset(repo_id):
    """Load a Hub dataset's train split, tolerating unparseable card metadata.

    Some dataset cards declare feature types a given `datasets` version can't
    parse (e.g. grug-think declares a 'Json' feature). That errors while reading
    the *card* — before any rows load. Fall back to reading the raw data files
    directly, which infers the schema from the data and skips the card entirely.
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo_id, split="train")
    except Exception as e:
        print(f"Standard load failed ({type(e).__name__}: {e}).")
        print("Falling back to reading raw data files from the Hub...")
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        candidates = fs.glob(f"datasets/{repo_id}/**/*.jsonl") or \
            fs.glob(f"datasets/{repo_id}/**/*.parquet")
        if not candidates:
            raise
        # Prefer files that look like the train split (avoid heldout/rl/eval files).
        train_files = [f for f in candidates if "train" in f.rsplit("/", 1)[-1].lower()]
        chosen = train_files or candidates
        builder = "json" if chosen[0].endswith(".jsonl") else "parquet"
        print(f"Loading {len(chosen)} file(s) with the '{builder}' builder: {chosen}")
        return load_dataset(builder, data_files=[f"hf://{f}" for f in chosen], split="train")


def build_dataset(args, tokenizer):
    from datasets import load_dataset

    if os.path.exists(args.dataset):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_hub_dataset(args.dataset)

    if args.subsample and 0 < args.subsample < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.subsample))

    def render(example):
        messages = normalize_messages(example["messages"])
        tools = example.get("tools") or None
        try:
            text = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = ""  # skip rows the template can't render
        return {"text": text}

    ds = ds.map(render, remove_columns=ds.column_names)
    ds = ds.filter(lambda e: len(e["text"]) > 0)
    # Drop rows that would be truncated, so we never train on half a tool call.
    ds = ds.filter(lambda e: len(tokenizer(e["text"]).input_ids) <= args.max_seq_len)
    return ds


def main():
    args = parse_args()

    from unsloth import FastLanguageModel

    # 1. Load the base model in 4-bit. Unsloth returns a normal PEFT-ready model.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )

    # 2. Build the dataset (handles tool rendering, subsampling, length filtering).
    dataset = build_dataset(args, tokenizer)
    print(f"Prepared {len(dataset)} training examples.")

    if args.preview:
        print("\n===== First rendered example =====\n")
        print(dataset[0]["text"])
        print("\n(preview only — exiting before training)")
        return

    # 3. Attach LoRA adapters — only these small matrices get trained.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # 4. Configure and build the trainer.
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to=args.report_to,
        ),
    )

    # 5. Train only on the assistant's turns (its <think> reasoning + tool calls +
    #    final replies), masking the user prompts and tool-result turns so the
    #    model learns to *produce* those, not to echo them.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    # 6. Save the LoRA adapter (small — just the trained matrices).
    model.save_pretrained(f"{args.output_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_adapter")
    print(f"\nDone! LoRA adapter saved to {args.output_dir}/lora_adapter")
    print("Test it with:  python inference.py")


if __name__ == "__main__":
    main()
