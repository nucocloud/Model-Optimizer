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
import re
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

    with open(config_path) as f:
        lora_config_dict = json.load(f)
    lora_sd = load_file(weights_path)
    print(f"Loaded {len(lora_sd)} LoRA tensors from {lora_dir}")

    # Strip any .default. segment from keys for compatibility across peft versions
    # e.g., lora_A.default.weight -> lora_A.weight
    lora_sd = {re.sub(r"\.default\.", ".", k): v for k, v in lora_sd.items()}

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

        with warnings.catch_warnings():
            # peft 0.18+ emits a spurious "missing adapter keys" warning
            # even when weights load correctly — suppress it.
            warnings.filterwarnings("ignore", message=".*missing adapter keys.*")
            model = PeftModel.from_pretrained(model, tmp_path)

        # Verify at least one LoRA weight is non-zero
        lora_norms = [v.norm().item() for k, v in model.state_dict().items() if ".lora_A." in k]
        if not lora_norms or all(n == 0 for n in lora_norms):
            raise RuntimeError("LoRA weights are all zero — adapter loading failed.")
        print(
            f"  Loaded {len(lora_norms)} LoRA-A matrices (mean norm={sum(lora_norms) / len(lora_norms):.4f})."
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
