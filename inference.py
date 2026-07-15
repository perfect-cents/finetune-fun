"""
Chat with your freshly finetuned adapter.

    python inference.py --adapter outputs/lora_adapter

Loads the base model + your LoRA adapter and runs an interactive prompt so you
can hear your model talk in its new voice. Ctrl-C or type 'quit' to exit.
"""

import argparse

from transformers import TextStreamer
from unsloth import FastLanguageModel


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a finetuned Qwen LoRA adapter.")
    p.add_argument("--adapter", default="outputs/lora_adapter")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--system",
        default="You are a helpful assistant. Think step by step inside <think> tags "
        "before answering.",
        help="System prompt. Note: grug-think is a tool-use dataset, so the model's "
        "grug-style reasoning shows up in <think> tags. To exercise real function "
        "calling you'd pass tool definitions via the chat template (see README).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Loading the adapter path also pulls the base model it was trained on.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # ~2x faster generation

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("Chatting with your model. Type 'quit' to exit.\n")

    while True:
        try:
            user_msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_msg.lower() in {"quit", "exit"}:
            break
        if not user_msg:
            continue

        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_msg},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        print("bot> ", end="")
        model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            streamer=streamer,
        )
        print()


if __name__ == "__main__":
    main()
