"""
Merge the LoRA adapter into the base model, quantize to GGUF, and generate an
Ollama Modelfile so you can run the finetuned model locally on your Mac.

Run this on the RunPod GPU box AFTER training (it needs the base model + Unsloth):

    python export_gguf.py --adapter outputs/lora_adapter --quant q4_k_m

Output lands in outputs/gguf/ :
    *.gguf        the quantized model (download this to your Mac)
    Modelfile     ready-to-use, with FROM already pointing at the .gguf

Then, on your Mac (see README):
    ollama create grug -f Modelfile
    ollama run grug
"""

import quiet  # noqa: E402  — must precede transformers/unsloth to silence import spam

import argparse
import glob
import os

from unsloth import FastLanguageModel

quiet.hush()

# A minimal ChatML template — correct for chatting with Qwen and seeing the grug
# <think> reasoning. (Full tool-calling through Ollama needs the tools-aware
# template; see the README note.)
MODELFILE_TEMPLATE = '''FROM ./{gguf_name}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7

SYSTEM """You are a helpful assistant. Think step by step inside <think> tags before answering."""
'''


def main():
    p = argparse.ArgumentParser(description="Export a finetuned adapter to GGUF for Ollama.")
    p.add_argument("--adapter", default="outputs/lora_adapter")
    p.add_argument("--output-dir", default="outputs/gguf")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument(
        "--quant",
        default="q4_k_m",
        help="GGUF quantization. q4_k_m = small+fast (good default for a Mac); "
        "q8_0 = larger, higher quality; f16 = full precision, biggest.",
    )
    args = p.parse_args()

    # Load the finetuned adapter (pulls the base model it was trained on).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
    )

    # Merge LoRA into the base weights and quantize to GGUF. Unsloth builds
    # llama.cpp under the hood the first time — this step can take a few minutes.
    print(f"Exporting GGUF ({args.quant}) to {args.output_dir}/ ...")
    model.save_pretrained_gguf(
        args.output_dir, tokenizer, quantization_method=args.quant
    )

    # Write a ready-to-use Modelfile pointing at whatever .gguf was produced.
    gguf_files = sorted(glob.glob(os.path.join(args.output_dir, "*.gguf")))
    if not gguf_files:
        raise SystemExit(f"No .gguf file found in {args.output_dir} — export failed.")
    gguf_name = os.path.basename(gguf_files[-1])
    modelfile_path = os.path.join(args.output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(MODELFILE_TEMPLATE.format(gguf_name=gguf_name))

    print(f"\nDone!")
    print(f"  GGUF:      {gguf_files[-1]}")
    print(f"  Modelfile: {modelfile_path}")
    print("\nDownload the outputs/gguf/ folder to your Mac, then:")
    print("  cd outputs/gguf && ollama create grug -f Modelfile && ollama run grug")


if __name__ == "__main__":
    main()
