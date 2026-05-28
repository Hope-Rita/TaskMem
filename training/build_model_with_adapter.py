import argparse
import os
import shutil

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText


CONFIG_FILES_TO_COPY = [
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject a per-layer zero-initialised `adapter.steer` "
                    "parameter into every transformer layer of a Qwen3-VL "
                    "checkpoint, matching the dtype of an existing "
                    "reference weight."
    )
    parser.add_argument("--src_path", type=str, required=True,
                        help="HF checkpoint to load (will be augmented)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="destination directory for the augmented checkpoint")
    parser.add_argument("--tokenizer_src", type=str, default=None,
                        help="directory to copy tokenizer/preprocessor json "
                             "files from. Defaults to --src_path.")
    parser.add_argument("--steer_dim", type=int, default=2048,
                        help="size of the per-layer steer vector")
    parser.add_argument("--reference_layer", type=int, default=31,
                        help="layer whose v_proj.weight dtype is used to "
                             "initialise the new steer parameters")
    parser.add_argument("--num_threads", type=int, default=8,
                        help="torch.set_num_threads value")
    parser.add_argument("--max_shard_size", type=str, default="5GB")
    return parser.parse_args()


def add_steer_parameters(args):
    os.makedirs(args.save_path, exist_ok=True)

    config = AutoConfig.from_pretrained(args.src_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.src_path,
        config=config,
        device_map="auto",
        torch_dtype=None,
    )

    reference_param = model.language_model.layers[args.reference_layer].self_attn.v_proj.weight
    reference_dtype = reference_param.dtype
    print(f"reference dtype @ layer {args.reference_layer}.v_proj: {reference_dtype}")

    num_layers = len(model.language_model.layers)
    for layer_num in range(num_layers):
        layer_module = model.language_model.layers[layer_num]
        steer_param = nn.Parameter(torch.zeros(args.steer_dim, dtype=reference_dtype))
        if not hasattr(layer_module, "adapter"):
            layer_module.adapter = nn.Module()
        layer_module.adapter.steer = steer_param
        print(f"layer {layer_num}: added adapter.steer (dim={args.steer_dim}, dtype={reference_dtype})")

    model.save_pretrained(
        args.save_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer_src = args.tokenizer_src or args.src_path
    for file in CONFIG_FILES_TO_COPY:
        src = os.path.join(tokenizer_src, file)
        dst = os.path.join(args.save_path, file)
        if os.path.exists(src):
            shutil.copy(src, dst)

    print(f"saved to {args.save_path}")


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    add_steer_parameters(args)
