# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge LoRA weights from an exported EAGLE checkpoint into the base model and save.

Usage:
    python merge_lora.py \
        --base_model_path /path/to/original/base/model \
        --exported_lora_dir /path/to/exported/eagle/checkpoint \
        --output_path /path/to/merged/output

The exported checkpoint (from export_hf_checkpoint.py) contains
lora_adapter_model.safetensors and lora_adapter_config.json. This script
loads the original base model, applies the trained LoRA adapters, merges
them into the base weights, and saves the fused model + tokenizer.
"""

import argparse
import json
from pathlib import Path

from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights from an exported EAGLE checkpoint into the base model."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the original base model (HF model name or local path).",
    )
    parser.add_argument(
        "--exported_lora_dir",
        type=str,
        required=True,
        help="Path to the exported EAGLE checkpoint containing lora_adapter_model.safetensors.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory to save the merged (fused) base model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lora_dir = Path(args.exported_lora_dir)

    # Verify exported files exist
    config_path = lora_dir / "lora_adapter_config.json"
    weights_path = lora_dir / "lora_adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected lora_adapter_config.json and lora_adapter_model.safetensors "
            f"in {lora_dir}. Run export_hf_checkpoint.py first."
        )

    # The exported LoRA keys already use PeftModel-compatible format
    # (e.g., model.layers.0.self_attn.q_proj.lora_A.default.weight).
    with open(config_path) as f:
        lora_config_dict = json.load(f)
    lora_sd = load_file(weights_path)
    print(f"Loaded {len(lora_sd)} LoRA tensors from {lora_dir}")

    # Prepare a temporary adapter directory that PeftModel can load
    import tempfile

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        save_file(lora_sd, tmp_path / "adapter_model.safetensors")
        # Write adapter_config.json for PeftModel
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(lora_config_dict, f)

        # Load the original base model
        print(f"Loading base model from {args.base_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, torch_dtype="auto", device_map="cpu", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

        # Load LoRA adapter via PeftModel and merge
        print("Loading and merging LoRA adapter...")
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PeftModel.from_pretrained(model, tmp_path)

        missing_warnings = [w for w in caught if "missing adapter keys" in str(w.message)]
        if missing_warnings:
            raise RuntimeError(
                f"LoRA weights failed to load — missing adapter keys. "
                f"Re-export the checkpoint with the latest export_hf_checkpoint.py.\n"
                f"{missing_warnings[0].message}"
            )
        model = model.merge_and_unload()

    print("LoRA merged successfully.")

    # Save
    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Done! Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
