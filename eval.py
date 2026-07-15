"""
Measure whether the finetuned model actually learned grug-think's behavior:
terse `<think>` reasoning followed by a correctly-formatted, valid tool call.

    python eval.py --adapter outputs/lora_adapter

    # Compare against the un-finetuned base model on the same prompts:
    python eval.py --adapter outputs/lora_adapter --compare-base unsloth/Qwen2.5-3B-Instruct

Plain chat won't exercise what this dataset teaches, so each test case supplies
real tool definitions and a user request, then we check the generated output for:
  1. a <think>...</think> reasoning block,
  2. a <tool_call>...</tool_call> block containing valid JSON,
  3. a tool name that actually exists in that case's tool list.
"""

import argparse
import json
import re

from unsloth import FastLanguageModel

# Held-out prompts the model never trained on. Each has its own tool menu.
TEST_CASES = [
    {
        "tools": [{
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
        }],
        "user": "What's the weather like in Tokyo right now?",
        "expect_tool": "get_weather",
    },
    {
        "tools": [{
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount from one currency to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        }],
        "user": "How much is 250 US dollars in euros?",
        "expect_tool": "convert_currency",
    },
    {
        "tools": [{
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
        }],
        "user": "Email alice@example.com to say the meeting is moved to 3pm.",
        "expect_tool": "send_email",
    },
]

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def generate(model, tokenizer, tools, user_msg, max_new_tokens):
    messages = [{"role": "user", "content": user_msg}]
    inputs = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Only decode the newly generated tokens, not the prompt.
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def score(output, expected_tool, tool_names):
    """Return (has_think, valid_tool_call, correct_tool, think_text)."""
    think_match = THINK_RE.search(output)
    has_think = think_match is not None
    think_text = think_match.group(1).strip() if think_match else ""

    valid_tool_call = False
    correct_tool = False
    call_match = TOOL_CALL_RE.search(output)
    if call_match:
        try:
            call = json.loads(call_match.group(1))
            name = call.get("name")
            valid_tool_call = "arguments" in call and name in tool_names
            correct_tool = name == expected_tool
        except json.JSONDecodeError:
            pass
    return has_think, valid_tool_call, correct_tool, think_text


def evaluate(label, model, tokenizer, max_new_tokens):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    totals = {"think": 0, "valid": 0, "correct": 0}
    for i, case in enumerate(TEST_CASES, 1):
        tool_names = {t["function"]["name"] for t in case["tools"]}
        output = generate(model, tokenizer, case["tools"], case["user"], max_new_tokens)
        has_think, valid, correct, think_text = score(output, case["expect_tool"], tool_names)
        totals["think"] += has_think
        totals["valid"] += valid
        totals["correct"] += correct

        print(f"\n[{i}] {case['user']}")
        print(f"    <think>: {'✓' if has_think else '✗'}"
              f"   valid tool_call: {'✓' if valid else '✗'}"
              f"   right tool ({case['expect_tool']}): {'✓' if correct else '✗'}")
        if think_text:
            print(f"    reasoning: {think_text[:120]}")
        print(f"    raw: {output.strip()[:220]}")

    n = len(TEST_CASES)
    print(f"\n  SUMMARY ({label}): "
          f"<think> {totals['think']}/{n}   "
          f"valid call {totals['valid']}/{n}   "
          f"right tool {totals['correct']}/{n}")
    return totals


def load(model_name, max_seq_len):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_len, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description="Eval a grug-think finetune on tool use.")
    p.add_argument("--adapter", default="outputs/lora_adapter")
    p.add_argument("--compare-base", default=None,
                   help="Optional base model id to run the same prompts against for comparison.")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    args = p.parse_args()

    model, tokenizer = load(args.adapter, args.max_seq_len)
    evaluate("FINETUNED", model, tokenizer, args.max_new_tokens)

    if args.compare_base:
        base_model, base_tok = load(args.compare_base, args.max_seq_len)
        evaluate("BASE (un-finetuned)", base_model, base_tok, args.max_new_tokens)
        print("\nCompare the two summaries: the finetune should produce grug <think> "
              "reasoning and clean tool calls more consistently than the base model.")


if __name__ == "__main__":
    main()
