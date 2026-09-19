from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np


def parse_molmo_points(text: str) -> list[tuple[float, float]]:
    xs = {int(i): float(v) for i, v in re.findall(r'x(\d+)="([^"]+)"', text)}
    ys = {int(i): float(v) for i, v in re.findall(r'y(\d+)="([^"]+)"', text)}
    return [(xs[i], ys[i]) for i in sorted(xs.keys() & ys.keys()) if 0 <= xs[i] <= 100 and 0 <= ys[i] <= 100]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_grounding(
    image_dir: Path,
    output: Path,
    tasks: list[int],
    target_type: str,
    model_id: str,
    seed: int,
    max_new_tokens: int,
) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    from transformers.generation.utils import GenerationMixin

    if not hasattr(GenerationMixin, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, *args, **kwargs):
            if getattr(outputs, "past_key_values", None) is not None:
                return "past_key_values", outputs.past_key_values
            if getattr(outputs, "mems", None) is not None:
                return "mems", outputs.mems
            if getattr(outputs, "past_buckets_states", None) is not None:
                return "past_buckets_states", outputs.past_buckets_states
            return "past_key_values", None

        GenerationMixin._extract_past_from_model_output = _extract_past_from_model_output

    seed_everything(seed)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    records = []
    for task in tasks:
        image_path = image_dir / f"{task}.jpg"
        image = Image.open(image_path).convert("RGB")
        prompt = f"Point all {target_type}."
        inputs = processor.process(images=[image], text=prompt)
        inputs = {key: value.to(model.device).unsqueeze(0) for key, value in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate_from_batch(
                inputs,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    stop_strings="<|endoftext|>",
                    do_sample=False,
                ),
                tokenizer=processor.tokenizer,
            )
        runtime = time.perf_counter() - started
        generated = output_ids[0, inputs["input_ids"].size(1) :]
        raw_text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        points = parse_molmo_points(raw_text)
        records.append(
            {
                "task_id": task,
                "target_type": target_type,
                "prompt": prompt,
                "model_id": model_id,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "points_percent": points,
                "num_targets": len(points),
                "runtime_s": runtime,
                "raw_text": raw_text,
            }
        )
        output.write_text(json.dumps(records, indent=2))
        print(f"task={task:02d} targets={len(points):02d} runtime_s={runtime:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="1-30")
    parser.add_argument("--target-type", default="buildings")
    parser.add_argument("--model-id", default="cyan2k/molmo-7B-O-bnb-4bit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if "-" in args.tasks:
        lo, hi = map(int, args.tasks.split("-", 1))
        tasks = list(range(lo, hi + 1))
    else:
        tasks = [int(v) for v in args.tasks.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_grounding(args.image_dir, args.output, tasks, args.target_type, args.model_id, args.seed, args.max_new_tokens)


if __name__ == "__main__":
    main()
