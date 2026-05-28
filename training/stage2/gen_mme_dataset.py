import argparse
import json
import os
import random
import re


SUPPLEMENT_PROMPT = """

Additional Output Requirements:
{}, which will be used to answer questions similar to those listed below:
{}
You need to generate content to ensure that such similar questions can be answered."""

EPISODIC_INSTRUCTION = "You need to focus on generating some related descriptions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VideoMME stage-2 episodic data preparation. Three "
                    "subcommands: emit per-video src/main.py command lists "
                    "for memory generation (gen_cmds_test) or rollout "
                    "(gen_cmds_rollout), or pack cached memory jsons into "
                    "SFT-ready jsonl with pre-extracted video features "
                    "(pack_sft)."
    )
    parser.add_argument("--type", type=str, required=True,
                        choices=["gen_cmds_test", "gen_cmds_rollout", "pack_sft"])
    parser.add_argument("--video_info", type=str, required=True,
                        help="path to VideoMME video_info.json (per-task records)")
    parser.add_argument("--video_root", type=str, required=True,
                        help="directory containing {videoID}.mp4 source videos")
    parser.add_argument("--output_root", type=str, required=True,
                        help="root directory for per-video memory outputs "
                             "(forwarded to src/main.py as --output_folder)")
    parser.add_argument("--cmds_out", type=str,
                        help="gen_cmds_test: destination json file. "
                             "gen_cmds_rollout: destination directory "
                             "(files written as {task_type}_{bound}.json)")
    parser.add_argument("--sft_out_dir", type=str,
                        help="pack_sft: directory to write "
                             "{task_type}_episodic_{bound}.jsonl files")
    parser.add_argument("--update_position", type=str, default="[10,30,50]",
                        help="JSON list of cumulative example-count cutoffs")
    parser.add_argument("--model_type", type=str, default="qwen3_vl_vllm",
                        help="forwarded to src/main.py as --episodic_model")
    parser.add_argument("--episodic_model", type=str, default="",
                        help="forwarded to src/main.py as --episodic_model_path")
    parser.add_argument("--read_memory_tag", type=str, default="taskmem_ep_seed",
                        help="forwarded to src/main.py as --read_memory_tag")
    parser.add_argument("--write_memory_prefix", type=str, default="taskmem_ep",
                        help="prefix of --write_memory_tag, full tag becomes "
                             "{prefix}_{task_type}_{tag}")
    parser.add_argument("--tag", type=str, default="1",
                        help="suffix used in --write_memory_tag")
    parser.add_argument("--task_type", type=str,
                        help="pack_sft: optional filter to a single task_type")
    parser.add_argument("--processor_path", type=str,
                        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="pack_sft: HF hub id or local path of the VL processor")
    return parser.parse_args()


def _make_supplement(questions, instruction):
    if not questions:
        return ""
    quoted = [q.replace('"', '\\"') for q in questions]
    return SUPPLEMENT_PROMPT.format(instruction, "- " + "\n- ".join(quoted))


def _episodic_cmd(args, video_id, task_type, supplement_prompt, *, vl_model=False):
    parts = [
        "python3 src/main.py",
        f"--video_path={os.path.join(args.video_root, f'{video_id}.mp4')}",
        f"--output_folder={os.path.join(args.output_root, video_id)}",
        "--generate_episodic",
        f"--read_memory_tag={args.read_memory_tag}",
        f"--write_memory_tag={args.write_memory_prefix}_{task_type}_{args.tag}",
        f'--supplement_prompt="{supplement_prompt}"',
    ]
    if vl_model:
        parts.append(f"--episodic_model={args.model_type}")
        if args.episodic_model:
            parts.append(f"--episodic_model_path={args.episodic_model}")
    return " ".join(parts)


def gen_cmds_test(args, datas, update_position):
    example_flag, prev_val = [], 0
    for curr_bound in update_position:
        example_flag += [prev_val] * (curr_bound - prev_val)
        prev_val = curr_bound
    example_flag += [prev_val]

    tasks = []
    for task_type, task_data in datas.items():
        seen = set()
        for i, data in enumerate(task_data):
            if data["videoID"] in seen:
                continue
            seen.add(data["videoID"])
            example_num = example_flag[i] if i < len(example_flag) else example_flag[-1]
            candidates = [j["question"] for j in task_data[:example_num]]
            questions = random.sample(candidates, min(5, len(candidates)))
            supplement = _make_supplement(questions, EPISODIC_INSTRUCTION)
            tasks.append(_episodic_cmd(args, data["videoID"], task_type, supplement))

    assert args.cmds_out, "--cmds_out is required for --type gen_cmds_test"
    os.makedirs(os.path.dirname(os.path.abspath(args.cmds_out)) or ".", exist_ok=True)
    with open(args.cmds_out, "w") as f:
        json.dump(tasks, f, indent=4, ensure_ascii=False)


def gen_cmds_rollout(args, datas, update_position):
    assert args.cmds_out, "--cmds_out is required for --type gen_cmds_rollout"
    out_dir = args.cmds_out
    os.makedirs(out_dir, exist_ok=True)

    for curr_bound in update_position:
        for task_type, task_data in datas.items():
            cmds = []
            for data in task_data[:curr_bound]:
                candidates = [j["question"] for j in task_data[:curr_bound]]
                sampled = random.sample(candidates, min(4, len(candidates)))
                questions = [data["question"]] + sampled
                supplement = _make_supplement(questions, EPISODIC_INSTRUCTION)
                cmds.append(
                    _episodic_cmd(args, data["videoID"], task_type, supplement, vl_model=True)
                )
            out_path = os.path.join(out_dir, f"{task_type}_{curr_bound}.json")
            with open(out_path, "w") as f:
                json.dump({"episodic": cmds}, f, indent=4, ensure_ascii=False)


def pack_sft(args, datas, update_position):
    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    assert args.sft_out_dir, "--sft_out_dir is required for --type pack_sft"
    os.makedirs(args.sft_out_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor_path)
    prefix = "which will be used to answer questions similar to those listed below:"
    suffix = "You need to generate content to ensure that such similar questions can be answered."
    pattern = re.escape(prefix) + r"(.*?)" + re.escape(suffix)

    for curr_bound in update_position:
        for task_type, task_data in datas.items():
            if args.task_type and task_type != args.task_type:
                continue
            out_path = os.path.join(
                args.sft_out_dir, f"{task_type}_episodic_{curr_bound}.jsonl"
            )
            with open(out_path, "w") as fout:
                for data in tqdm(task_data[:curr_bound], desc=f"{task_type}@{curr_bound}"):
                    supplement_prompts = None
                    idx = 0
                    while True:
                        clip_dir = os.path.join(data["memory_path"], str(idx))
                        if not os.path.exists(clip_dir):
                            break
                        mem_file = os.path.join(
                            clip_dir,
                            f"{data['videoID']}_{idx}_{args.write_memory_prefix}_{task_type}_{args.tag}.json",
                        )
                        memory = json.load(open(mem_file))
                        for k, v in memory.items():
                            if k != "episodic":
                                continue
                            if supplement_prompts is None:
                                extracted = re.findall(
                                    pattern, v["input"][-1]["text"], flags=re.DOTALL
                                )
                                supplement_prompts = extracted[0] if extracted else ""
                            messages = [{"role": "user", "content": v["input"]}]
                            text = processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            video_save_path = os.path.join(
                                clip_dir,
                                f"{data['videoID']}_{idx}_{args.write_memory_prefix}_{task_type}_{args.tag}_video.pt",
                            )
                            if not os.path.exists(video_save_path):
                                _, videos, _ = process_vision_info(
                                    messages,
                                    image_patch_size=processor.image_processor.patch_size,
                                    return_video_kwargs=True,
                                    return_video_metadata=True,
                                )
                                torch.save(videos[0], video_save_path)
                            row = {
                                "id": f"{data['videoID']}_{idx}",
                                "type": k,
                                "input": v["input"],
                                "output": v["output"],
                                "text": text,
                                "images": None,
                                "videos": video_save_path,
                                "video_kwargs": {"do_sample_frames": False},
                                "supplement_prompts": supplement_prompts,
                            }
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        idx += 1


def main():
    args = parse_args()
    update_position = json.loads(args.update_position)
    with open(args.video_info) as f:
        datas = json.load(f)

    if args.type == "gen_cmds_test":
        gen_cmds_test(args, datas, update_position)
    elif args.type == "gen_cmds_rollout":
        gen_cmds_rollout(args, datas, update_position)
    elif args.type == "pack_sft":
        pack_sft(args, datas, update_position)


if __name__ == "__main__":
    main()
