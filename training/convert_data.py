import argparse
import json
import uuid
from collections import defaultdict

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a TaskMem episodic training jsonl into a "
                    "parquet file consumable by the verl training loop. "
                    "Each line is either a single record (dict) for "
                    "SFT-style rollout or a list of records for DPO-style "
                    "preference pairs."
    )
    parser.add_argument("--src_json", type=str, required=True,
                        help="input .jsonl path")
    parser.add_argument("--tgt_parquet", type=str, default=None,
                        help="output .parquet path (defaults to src_json "
                             "with .jsonl swapped for .parquet)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tgt_parquet is None:
        args.tgt_parquet = args.src_json.replace(".jsonl", ".parquet")
    assert args.src_json != args.tgt_parquet, "src and tgt must differ"

    datas = defaultdict(list)
    with open(args.src_json) as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, dict):
                datas["id"].append(data["id"])
                datas["type"].append(data["type"])
                datas["videos"].append([data["input"][1]])
                datas["input"].append(data["input"])
                datas["prompt"].append([{
                    "content": data["input"][0]["text"] + "<video>" + data["input"][2]["text"],
                    "role": "user",
                }])
                datas["response"].append(data.get("output", ""))
                datas["need_rollout"].append(True)
            elif isinstance(data, list):
                uid = str(uuid.uuid4())
                for item in data:
                    datas["uid"].append(uid)
                    datas["id"].append(item["id"])
                    datas["type"].append(item["type"])
                    datas["videos"].append([item["input"][1]])
                    datas["input"].append(item["input"])
                    datas["prompt"].append([{
                        "content": item["input"][0]["text"] + "<video>" + item["input"][2]["text"],
                        "role": "user",
                    }])
                    datas["response"].append(item["response"])
                    datas["reward"].append(item["reward"])
                    datas["task_score"].append(item["task_score"])
                    datas["correctness"].append(item["correctness"])
                    datas["need_rollout"].append(False)
            else:
                raise NotImplementedError(f"Unsupported record type: {type(data)}")

    pd.DataFrame(datas).to_parquet(args.tgt_parquet, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
