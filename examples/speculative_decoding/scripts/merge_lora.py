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

from peft import LoraConfig
from peft.mapping import inject_adapter_in_model
from peft.tuners.lora import LoraLayer
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

    # Load LoRA config and weights from the exported checkpoint
    config_path = lora_dir / "lora_adapter_config.json"
    weights_path = lora_dir / "lora_adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected lora_adapter_config.json and lora_adapter_model.safetensors "
            f"in {lora_dir}. Run export_hf_checkpoint.py first."
        )

    with open(config_path) as f:
        lora_config_dict = json.load(f)
    lora_config = LoraConfig(
        **{k: v for k, v in lora_config_dict.items() if k in LoraConfig().to_dict()}
    )
    lora_sd = load_file(weights_path)
    print(
        f"Loaded LoRA config (rank={lora_config.r}, alpha={lora_config.lora_alpha}) "
        f"and {len(lora_sd)} tensors from {lora_dir}"
    )

    # Load the original base model
    print(f"Loading base model from {args.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype="auto", device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)

    # Inject LoRA adapters into the base model's transformer body
    # The exported keys are prefixed with "model." (the base_model_path in the EAGLE model)
    print("Injecting LoRA adapters...")
    inject_adapter_in_model(lora_config, model.model, adapter_name="default")

    # Strip the "model." prefix from exported keys to match the injected model.model namespace
    prefix = "model."
    cleaned_sd = {k.removeprefix(prefix): v for k, v in lora_sd.items()}

    missing, unexpected = model.model.load_state_dict(cleaned_sd, strict=False)
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"WARNING: Missing LoRA keys: {lora_missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    print("LoRA weights loaded.")

    # Merge LoRA into base model weights and remove adapters
    print("Merging LoRA into base weights...")
    for module in model.model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
            module.delete_adapter("default")
    print("LoRA merged and removed.")

    # Save
    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Done! Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
