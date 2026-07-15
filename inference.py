"""
Chat with your finetuned adapter — with tools available and <think> reasoning
forced on, so you actually see the grug-think behavior the model was trained for.

    python inference.py --adapter outputs/lora_adapter

    # Plain chat, no tools, no forced thinking:
    python inference.py --no-tools --no-think

Why the defaults matter: grug-think reasons inside <think> tags when deciding
which tool to call. So we (1) pass a small example tool menu and (2) prefill the
"<think>" tag to force the model to start reasoning. Ctrl-C or 'quit' to exit.
"""

import quiet  # noqa: E402  — must precede transformers/unsloth to silence import spam

import argparse

from transformers import TextStreamer
from unsloth import FastLanguageModel

quiet.hush()

# A small example tool menu so the model has something to reason about / call.
EXAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. '12 * 7'"}},
                "required": ["expression"],
            },
        },
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a finetuned Qwen LoRA adapter.")
    p.add_argument("--adapter", default="outputs/lora_adapter")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--system", default="You are a helpful assistant with access to tools. "
                   "Reason about the request inside <think> tags, then act.")
    p.add_argument("--no-think", dest="force_think", action="store_false",
                   help="Don't prefill <think> (let the model decide whether to reason).")
    p.add_argument("--no-tools", dest="use_tools", action="store_false",
                   help="Don't offer any tools (plain chat).")
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # ~2x faster generation
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    tools = EXAMPLE_TOOLS if args.use_tools else None
    print("Chatting with your model"
          f"{' (tools available)' if tools else ''}"
          f"{', <think> forced' if args.force_think else ''}. Type 'quit' to exit.\n")

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
        # Render to text so we can prefill "<think>" onto the end of the prompt.
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        if args.force_think:
            prompt += "<think>"
        # add_special_tokens=False: the template already emits the special tokens.
        # Passing the attention_mask (returned here) avoids the pad/eos warning.
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Echo the prefilled tag so the streamed reasoning reads as one block.
        print("bot> " + ("<think>" if args.force_think else ""), end="", flush=True)
        model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
        print()


if __name__ == "__main__":
    main()
