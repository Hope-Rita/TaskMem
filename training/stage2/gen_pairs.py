import argparse
import base64
import copy
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from time import sleep
from typing import Any, Dict, List, Tuple

import openai
from json_repair import loads, repair_json
from transformers import AutoTokenizer


MAX_RETRIES = 5
DEFAULT_TEMPERATURE = 1e-6
DEFAULT_TIMEOUT = 480

MODEL_NAME_TEXT = os.environ.get("TASKMEM_TEXT_JUDGE", "gpt-4o-2024-11-20")
MODEL_NAME_VIDEO = os.environ.get("TASKMEM_VIDEO_JUDGE", "gemini-2.5-flash")

CONFIG_PATH = os.environ.get(
    "TASKMEM_API_CONFIG", os.path.join("configs", "api_config.json")
)
TOKENIZER_PATH = os.environ.get(
    "TASKMEM_TOKENIZER_PATH", "Qwen/Qwen3-VL-30B-A3B-Thinking"
)


with open(CONFIG_PATH) as _f:
    config = json.load(_f)

client: Dict[str, Any] = {}
for _model_name in config.keys():
    if isinstance(config[_model_name], list):
        client[_model_name] = [
            openai.AzureOpenAI(
                azure_endpoint=conf["azure_endpoint"],
                api_version=conf["api_version"],
                api_key=conf["api_key"],
            )
            for conf in config[_model_name]
        ]
    else:
        client[_model_name] = openai.AzureOpenAI(
            azure_endpoint=config[_model_name]["azure_endpoint"],
            api_version=config[_model_name]["api_version"],
            api_key=config[_model_name]["api_key"],
        )


USER_PROMPT_EPISODIC_SINGLE_VIDEO = """You are provided with a video, a description of its preceding segment, and a generated candidate [Description] for the remaining portion.
Your task is to evaluate:
1. Whether the candidate description is factually accurate based only on visual content and subtitles (ignore audio).
2. Whether it connects coherently and naturally with the preceding description, without using transition words such as "continue".
For any spoken content, verify it solely against the displayed subtitles and disregard audio information.
Assign exactly one label:
1: Correct — The description that meets all of the above criteria.
0: Incorrect — Any description that fails to meet the above criteria.

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.
{{
    "correctness_rationale": "Short explanation for marking this description as 1 or 0",
    "correctness": 1 or 0 
}}

The description of the preceding segment:
{preceding_json}

The [Description] to verify:
{blocks_text}
""".strip()


USER_PROMPT_EPISODIC_TEXT = """
You are given the [Context] and a candidate description that are describing new events.

Your task is to evaluate whether the candidate description satisfies the following conditions.

Return label=0 if any condition is satisfied, else 1:
(1) The description repeats any atomic fact already present in the [Context].
(2) It includes any mention of bounding boxes, coordinates, or detection boxes (e.g., "bounding box", "bbox", "x1,y1,x2,y2", "rectangle box around").
(3) It contains meta phrases like: "subtitles said", "the subtitles say", "subtitle reads", "subtitle says", or "according to the subtitles".
(4) The quoted speech contains transcript-style speaker labels like "<face_id> says "<face_id>: Good"" inside quoted dialogue.
(5) It includes conclusion-based or context-setting statements such as "this video ends with..." or "based on previous videos".

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.

{{
    "label_rationale": "Short explanation for marking this description as 1 or 0",
    "label": 1 or 0
}}

[Context]:
{preceding_json}

candidate description to verify:
{blocks_text}

""".strip()


HELPFULNESS_EPISODIC_PROMPT = """
You are given two [Description] and some example questions.
Based on the focus of the example questions, your task is to evaluate which description contains information that would be more useful for answering similar questions.
Output the ID of the more useful description. If both descriptions are equally useful (a tie), output -1.
- A set of example questions: {example_questions}
- Two [Description]:
{blocks_text}
Return exactly one JSON object:
{{
    "more_useful_rationale": "Briefly introduce the reasons for making this judgment",
    "more_useful": "ID of the more useful description or -1"
}}
"""


prompts = {
    ("episodic", "correctness"): USER_PROMPT_EPISODIC_SINGLE_VIDEO,
    ("episodic", "label"): USER_PROMPT_EPISODIC_TEXT,
    ("episodic", "helpfulness"): HELPFULNESS_EPISODIC_PROMPT,
}


_tokenizer = None
_tok_lock = threading.Lock()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _tok_lock:
            if _tokenizer is None:
                _tokenizer = AutoTokenizer.from_pretrained(
                    TOKENIZER_PATH,
                    trust_remote_code=True,
                    use_fast=True,
                )
    return _tokenizer


def count_tokens_texts(texts):
    tok = get_tokenizer()
    enc = tok(
        [t if isinstance(t, str) else str(t) for t in texts],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    return [len(ids) for ids in enc["input_ids"]]


def get_response_with_retry(model, messages, timeout=30):
    for _ in range(MAX_RETRIES):
        try:
            if isinstance(client[model], list):
                selected_model = random.choice(client[model])
            else:
                selected_model = client[model]
            if model in ["gemini-2.5-flash", "gemini-2.5-pro"]:
                extra_body = {
                    "thinking": {"include_thoughts": True, "budget_tokens": 128}
                }
                response = selected_model.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    timeout=timeout,
                    extra_body=extra_body,
                    max_tokens=8192,
                )
            else:
                response = selected_model.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    timeout=timeout,
                    max_tokens=8192,
                )
            return response.choices[0].message.content
        except Exception as e:
            print("Failed to get response:", e)
            sleep(5)
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def generate_messages(inputs):
    messages = [
        {"role": "system", "content": "You are an expert in video understanding."}
    ]
    content = []
    for inp in inputs:
        if inp["type"] == "text":
            content.append(inp)
        elif inp["type"] == "video":
            base64_video = base64.b64encode(open(inp["video"], "rb").read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/mp4;base64,{base64_video}"},
            })
        else:
            raise ValueError(f"Invalid input type: {inp['type']}")
    messages.append({"role": "user", "content": content})
    return messages


def build_blocks(groups, mode, task_type):
    assert len(groups) == 1, f"build_blocks expects exactly 1 group, got {len(groups)}"
    descs = groups[0][mode]
    if not descs:
        return "", None
    return descs[0], None


def _invalid_face_tag(desc: str) -> bool:
    if re.search(r"<face>", desc, re.IGNORECASE):
        return True
    suspicious_pattern = re.compile(r"face[\s_]*\d+", re.IGNORECASE)
    for m in suspicious_pattern.finditer(desc):
        s, e = m.span()
        token = m.group()
        if token != f"face_{token.split('_')[-1]}" or not re.fullmatch(r"face_\d+", token):
            if not re.fullmatch(r"face_\d+", token):
                return True
        has_brackets = (
            s > 0 and desc[s - 1] == "<" and e < len(desc) and desc[e] == ">"
        )
        if not has_brackets:
            return True
    return False


def _validate_score_value(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v.strip())
            return True
        except Exception:
            return False
    return False


def _parse_llm_response(responses, valid_num, expected_idx, task_type):
    parsed = loads(repair_json(responses.strip().strip("`json").strip()))
    score_key = task_type

    if not isinstance(parsed, dict):
        raise ValueError("Missing or invalid JSON object.")

    if valid_num is None:
        if score_key not in parsed:
            raise ValueError(f"missing '{score_key}' in response")
        if expected_idx is None:
            if not _validate_score_value(parsed[score_key]):
                raise ValueError(f"not int value: {parsed}")
            return {
                score_key: {"1": parsed[score_key]},
                f"{score_key}_rationale": {"1": parsed.get(f"{score_key}_rationale", "")},
            }
        idx_dict = parsed.get(score_key)
        if set(idx_dict.keys()) != expected_idx:
            raise ValueError(
                f"idx mismatch. expected={expected_idx}, got={set(idx_dict.keys())}"
            )
        for k, v in idx_dict.items():
            if not _validate_score_value(v):
                raise ValueError(
                    f"invalid {score_key}[{k}] value type: {type(v)} {str(v)[:80]}"
                )
        return parsed

    idx_dict = parsed.get(score_key)
    if set(idx_dict.keys()) != valid_num:
        raise ValueError(
            f"idx mismatch. expected={valid_num}, got={set(idx_dict.keys())}"
        )
    for k, v in idx_dict.items():
        if not _validate_score_value(v):
            raise ValueError(
                f"invalid {score_key}[{k}] value type: {type(v)} {str(v)[:80]}"
            )
    return parsed


def _process_task(task_inputs, task_type, valid_num, expected_idx, model_name, timeout):
    last_responses = ""
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            messages = generate_messages(task_inputs)
            responses = get_response_with_retry(model_name, messages, timeout)
            last_responses = responses
            llm_responses = _parse_llm_response(responses, valid_num, expected_idx, task_type)
            return {task_type: llm_responses, f"{task_type}_responses": responses}
        except Exception as e:
            last_error = e
    print(f"[EvalLLM] {task_type} failed: {last_error}, last response: {last_responses}")
    return {task_type: {}, f"{task_type}_responses": last_responses}


def _eval_one_group(group, mode, task_type, downstream_prompt, timeout=DEFAULT_TIMEOUT):
    tmpl = prompts.get((mode, task_type))
    if not tmpl:
        return {task_type: {}, f"{task_type}_responses": ""}

    gid = group["group_id"]
    descs = group.get(mode, [])
    if not descs:
        return {
            task_type: {gid: {f"{task_type}_rationale": {}, task_type: {}}},
            f"{task_type}_responses": "",
        }

    blocks_text, expected_idx = build_blocks([group], mode, task_type)
    if not blocks_text:
        return {task_type: {}, f"{task_type}_responses": ""}

    prompt = tmpl.format(
        preceding_json=group.get("preceding_description", ""),
        example_questions=downstream_prompt,
        blocks_text=blocks_text,
    )

    if task_type == "correctness":
        inputs = [
            {"type": "video", "video": group["video_path"]},
            {"type": "text", "text": prompt},
        ]
        model_name = MODEL_NAME_VIDEO
    else:
        inputs = [{"type": "text", "text": prompt}]
        model_name = MODEL_NAME_TEXT

    result = _process_task(inputs, task_type, expected_idx, None, model_name, timeout)
    parsed = result.get(task_type, {})
    return {
        task_type: {gid: parsed},
        f"{task_type}_responses": result.get(f"{task_type}_responses", ""),
    }


def _eval_llm(groups, mode, task_type, downstream_prompts, timeout=DEFAULT_TIMEOUT):
    tmpl = prompts.get((mode, task_type))
    if not tmpl:
        return {task_type: {}}

    merged_results = {}
    merged_responses = {}
    worker_num = min(len(groups), 4)

    def run_one_group(g):
        gid = g["group_id"]
        try:
            result = _eval_one_group(g, mode, task_type, downstream_prompts, timeout=timeout)
            return (
                gid,
                result.get(task_type, {}).get(gid, {}),
                result.get(f"{task_type}_responses", ""),
            )
        except Exception as e:
            print(f"[EvalLLM] {task_type} for group {gid} failed: {e}")
            return gid, {}, ""

    with ThreadPoolExecutor(max_workers=worker_num) as ex:
        futures = [ex.submit(run_one_group, g) for g in groups]
        for fut in as_completed(futures):
            gid, one_result, one_response = fut.result()
            merged_results[str(gid)] = one_result
            merged_responses[str(gid)] = one_response

    return {
        task_type: merged_results,
        f"{task_type}_responses": json.dumps(merged_responses, ensure_ascii=False, indent=2),
    }


def _eval_episodic_label(groups, mode, task_type, downstream_prompts="", timeout=DEFAULT_TIMEOUT):
    tmpl = prompts.get((mode, task_type))
    results = {}
    responses = {}
    for g in groups:
        gid = str(g.get("group_id"))
        descs = g.get(mode, []) or []
        if descs and descs[0]:
            prompt = tmpl.format(
                preceding_json=groups[0].get("preceding_description", ""),
                blocks_text=descs[0],
            )
            inputs = [{"type": "text", "text": prompt}]
            try:
                results_g = _process_task(inputs, task_type, None, None, MODEL_NAME_TEXT, timeout)
                results[gid] = results_g.get(task_type, {})
                responses[gid] = results_g.get(f"{task_type}_responses", "")
            except Exception as e:
                print(f"[EvalLLM] {task_type} for group {gid} failed: {e}")
                results[gid] = {f"{task_type}_rationale": {}, task_type: {}}
        else:
            results[gid] = {f"{task_type}_rationale": {}, task_type: {}}
    return {
        task_type: results,
        f"{task_type}_responses": json.dumps(responses, ensure_ascii=False, indent=2),
    }


def _run_task(groups, mode, task_type, downstream_prompts, timeout=DEFAULT_TIMEOUT):
    if task_type == "correctness":
        return _eval_llm(groups, mode, task_type, downstream_prompts, timeout)
    if task_type == "label":
        return _eval_episodic_label(groups, mode, task_type, downstream_prompts, timeout)
    raise ValueError(f"Unsupported task_type: {task_type}")


def _evaluate_all_tasks(groups, mode, downstream_prompts, timeout=240):
    tasks = ["correctness", "label"]
    task_results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        future_to_type = {
            ex.submit(_run_task, groups, mode, task_type, downstream_prompts, timeout): task_type
            for task_type in tasks
        }
        for fut in as_completed(future_to_type):
            task_type = future_to_type[fut]
            try:
                task_results[task_type] = fut.result()
            except Exception as e:
                print(f"[EvalTask] {task_type} failed: {e}")
                task_results[task_type] = None
    return [task_results[t] if task_results[t] is not None else {} for t in tasks]


def _flatten_results(results):
    merged: Dict[str, Any] = {"global": {}}
    for r in results:
        for k, v in r.items():
            if isinstance(v, dict):
                for gid, data in v.items():
                    merged.setdefault(gid, {}).update(data)
            else:
                merged["global"][k] = v
    return merged


def normalize_descs(out_obj) -> List[str]:
    if out_obj is None:
        return []
    if isinstance(out_obj, str):
        return [out_obj]
    if isinstance(out_obj, list):
        return out_obj
    return []


def _json_list_after_key(block: str, key_name: str) -> List[str]:
    m = re.search(rf"\[\s*{re.escape(key_name)}\s*\]\s*:\s*", block, flags=re.IGNORECASE)
    if not m:
        return []
    after = block[m.end():]
    start = after.find("[")
    if start == -1:
        return []
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(after[start:], start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    if end is None:
        return []
    candidate = after[start:end]
    try:
        arr = json.loads(candidate)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def _parse_section_inputs(section_input: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    texts = [it["text"] for it in section_input if it.get("type") == "text" and "text" in it]
    merged = "\n\n".join(texts)
    last_text = texts[-1] if texts else ""
    pre = (
        last_text
        .split("[Description of the preceding part]:")[1]
        .split("\n\n- Generate subsequent descriptions not")[0]
        .strip()
    )
    supp = _json_list_after_key(last_text, "Supplementary features")
    if (not supp) and ("[Supplementary features]" not in last_text):
        supp = _json_list_after_key(merged, "Supplementary features")
    return pre or "", supp or []


def correctness_computation(results, mode, groups, args):
    new_groups = []
    for g in groups:
        gid = str(g["group_id"])
        g["winner"] = 2
        descs = g[mode]
        m_count = len(descs)
        g["m_count"] = m_count
        if g["fail_parsing"]:
            g["num_correct"], g["num_wrong"] = 0, 0
            continue
        s_final_corr = [0] * m_count
        s_task = [0] * m_count
        results_i = results.get(gid, {})
        if "correctness" not in results_i or "label" not in results_i:
            continue
        s_corr = list(results[gid]["correctness"].values())
        s_label = list(results[gid]["label"].values())
        if len(s_corr) != m_count or len(s_label) != m_count:
            continue

        g["num_correct"], g["num_wrong"] = 0, 0
        g["correct_list"] = [0] * m_count
        desc_token_lens = g["desc_token_lens"]

        for j, d in enumerate(descs):
            if (m_count > args.memory_length_threshold) and args.training:
                s_task[j] = -0.5
                continue
            too_long = desc_token_lens[j] > args.memory_token_threshold and args.training
            s_final_corr[j] = 0 if (
                _invalid_face_tag(d)
                or (not s_corr[j])
                or (not s_label[j])
                or too_long
            ) else 1
            s_task[j] = 0.5 if s_final_corr[j] else -0.5
            g["correct_list"][j] = s_final_corr[j]
            g["num_correct"] += s_final_corr[j]

        g["num_wrong"] = m_count - g["num_correct"]
        g["correctness"] = sum(s_final_corr)
        g["label"] = s_label
        g["ori_correctness"] = s_corr
        for key in ("correctness", "label"):
            g[f"cot_{key}"] = results[gid][f"{key}_rationale"]
        g["task_score"] = float(sum(s_task))
        new_groups.append(g)
    return new_groups


def load_groups_from_rollouts(json_path: str, mode: str, rollout: int, args):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert mode == "episodic", f"only episodic mode is supported, got {mode}"

    block = data.get(mode, {})
    inp = copy.deepcopy(block.get("input", []))
    pre, supp = _parse_section_inputs(inp)

    vids = [x for x in inp if x.get("type") == "video"]
    video_path_abs = vids[0].get("video", "") if vids else ""
    assert video_path_abs and os.path.isfile(video_path_abs), f"Video missing: {video_path_abs}"

    rollouts = block.get("rollout", []) or []
    groups = []
    for ridx, response in enumerate(rollouts[:rollout]):
        thinking_token = response.split("</think>")[0]
        thinking_token_lens = count_tokens_texts([thinking_token])[0]
        try:
            memory_desc = json.loads(
                response.split("</think>")[-1].strip().strip("```json").strip()
            )
            if (
                not isinstance(memory_desc, dict)
                or "description" not in memory_desc.keys()
                or len(memory_desc.keys()) > 1
                or len(memory_desc["description"]) == 0
            ):
                descs = []
            else:
                descs = [memory_desc["description"]]
        except Exception:
            descs = []

        desc_token_lens = count_tokens_texts(descs) if descs else []
        fail_parsing = (
            len(descs) == 0 or thinking_token_lens > args.thinking_token_threshold
        )

        groups.append({
            "type": mode,
            "group_id": ridx,
            mode: descs,
            "input": inp,
            "preceding_description": pre,
            "supplementary_features": supp,
            "video_path": video_path_abs,
            "fail_parsing": fail_parsing,
            "thinking_token_lens": thinking_token_lens,
            "desc_token_lens": desc_token_lens,
            "json_path": json_path,
            "rollout_response_text": response,
        })
    return groups


def _eval_helpfulness_episodic_pair(g0, g1, mode, downstream_prompts, timeout=DEFAULT_TIMEOUT):
    tmpl = prompts.get((mode, "helpfulness"))
    assert tmpl is not None

    d0 = (g0.get(mode, []) or [""])[0]
    d1 = (g1.get(mode, []) or [""])[0]
    if not d0 or not d1:
        return None

    blocks_text = json.dumps({"0": d0, "1": d1}, ensure_ascii=False, indent=4)
    prompt = tmpl.format(example_questions=downstream_prompts, blocks_text=blocks_text)
    task_inputs = [{"type": "text", "text": prompt}]
    score_key = "more_useful"
    result = _process_task(task_inputs, score_key, None, None, MODEL_NAME_TEXT, timeout)

    response = result.get(score_key)
    if not response:
        return None
    try:
        cand = int(response[score_key]["1"])
        return cand if cand in (0, 1) else None
    except Exception:
        return None


def choose_pairs_episodic_all(
    groups,
    mode,
    downstream_prompts,
    timeout=DEFAULT_TIMEOUT,
    max_consecutive_no_record=30,
):
    cand = []
    for g in groups:
        descs = g.get(mode, []) or []
        if g.get("fail_parsing"):
            continue
        if not descs or not descs[0]:
            continue
        cand.append(g)

    pair_inputs = list(combinations(cand, 2))
    if not pair_inputs:
        print("Total episodic pairs: 0")
        return None

    random.shuffle(pair_inputs)
    pairs_out = []
    consecutive_no_record = 0
    for g0, g1 in pair_inputs:
        try:
            win01 = _eval_helpfulness_episodic_pair(
                g0, g1, mode, downstream_prompts, timeout=timeout
            )
            if win01 is None or win01 == -1:
                consecutive_no_record += 1
                if consecutive_no_record >= max_consecutive_no_record:
                    break
                continue

            win10 = _eval_helpfulness_episodic_pair(
                g1, g0, mode, downstream_prompts, timeout=timeout
            )
            if win10 is None or win10 == -1:
                consecutive_no_record += 1
                if consecutive_no_record >= max_consecutive_no_record:
                    break
                continue

            winner01 = g0 if win01 == 0 else g1
            winner10 = g1 if win10 == 0 else g0
            if winner01["group_id"] != winner10["group_id"]:
                consecutive_no_record += 1
                print(
                    f"skip pair {g0['group_id']} vs {g1['group_id']} "
                    f"(inconsistent winner), consecutive_no_record={consecutive_no_record}"
                )
                if consecutive_no_record >= max_consecutive_no_record:
                    break
                continue

            better = winner01
            worse = g1 if better["group_id"] == g0["group_id"] else g0

            pair = [
                {
                    "reward": 1,
                    "response": better["rollout_response_text"],
                    "task_score": better.get("task_score"),
                    "correctness": better.get("correctness"),
                    "input": better["input"],
                },
                {
                    "reward": 0,
                    "response": worse["rollout_response_text"],
                    "task_score": worse.get("task_score"),
                    "correctness": worse.get("correctness"),
                    "input": worse["input"],
                },
            ]
            pairs_out.append(pair)
            print(
                f"saving pair: {better['group_id']} > {worse['group_id']} | "
                f"better_corr={better.get('correctness')} | "
                f"worse_corr={worse.get('correctness')} | "
                f"len_diff={abs(better['desc_token_lens'][0] - worse['desc_token_lens'][0])} | "
                f"total={len(pairs_out)}"
            )
            consecutive_no_record = 0
        except Exception as e:
            consecutive_no_record += 1
            print(
                f"Pair failed: {g0['group_id']} vs {g1['group_id']}, err={e}, "
                f"consecutive_no_record={consecutive_no_record}"
            )
            if consecutive_no_record >= max_consecutive_no_record:
                break

    print("Total episodic pairs:", len(pairs_out))
    return pairs_out if pairs_out else None


def process_one_record(items: Dict[str, Any], args):
    json_path = items["json_path"]
    supplement_prompts = items.get("downstream_prompt", "") or ""
    if not os.path.exists(json_path):
        print(f"[ProcessRecord] empty file: {json_path}")
        return None

    mode = "episodic"
    groups = load_groups_from_rollouts(json_path, mode=mode, rollout=args.rollout, args=args)
    result = _evaluate_all_tasks(
        groups, mode=mode, downstream_prompts=supplement_prompts, timeout=args.timeout
    )
    result_map = _flatten_results(result)
    new_groups = correctness_computation(result_map, mode, groups, args)
    return choose_pairs_episodic_all(new_groups, mode, supplement_prompts)


def main_groups(args):
    out_path = args.out_jsonl
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    example_questions = []
    with open(args.question_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                example_questions.append(line)

    video_id = os.path.basename(args.video_folder)
    sample_size = min(5, len(example_questions))

    tasks = []
    for t in range(args.start_time, args.end_time, args.clip_seconds):
        clip_idx = str(t // args.clip_seconds)
        tasks.append({
            "json_path": os.path.join(
                args.video_folder,
                clip_idx,
                f"{video_id}_{clip_idx}_{args.memory_tag}.json",
            ),
            "downstream_prompt": "\n- " + "\n- ".join(
                random.sample(example_questions, k=sample_size)
            ),
        })

    with open(out_path, "w", encoding="utf-8") as fw:
        for items in tasks:
            if not items:
                continue
            pair_or_pairs = process_one_record(items, args)
            if pair_or_pairs is None:
                continue

            to_write = []
            if isinstance(pair_or_pairs, list) and pair_or_pairs:
                if isinstance(pair_or_pairs[0], list):
                    to_write.extend(pair_or_pairs)
                else:
                    to_write.append(pair_or_pairs)
            else:
                to_write.append(pair_or_pairs)

            for pair in to_write:
                fw.write(json.dumps(pair, ensure_ascii=False) + "\n")
            fw.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-2 (episodic) pair generation. For each clip under "
                    "--video_folder, consume the cached rollout JSON, score "
                    "rollouts with LLM judges (correctness + label + pairwise "
                    "helpfulness), and write (winner, loser) preference pairs."
    )
    parser.add_argument("--video_folder", type=str, required=True,
                        help="root containing one subdirectory per clip index")
    parser.add_argument("--memory_tag", type=str, required=True,
                        help="suffix used in {video_id}_{clip_idx}_{memory_tag}.json")
    parser.add_argument("--start_time", type=int, required=True)
    parser.add_argument("--end_time", type=int, required=True)
    parser.add_argument("--clip_seconds", type=int, default=10,
                        help="clip length in seconds")
    parser.add_argument("--out_jsonl", type=str, default="./output.jsonl")
    parser.add_argument("--question_path", type=str, required=True,
                        help="text file with one example question per line")
    parser.add_argument("--rollout", type=int, required=True,
                        help="number of rollouts to consume per clip")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--training", type=int, default=1)
    parser.add_argument("--memory_length_threshold", type=int, default=5)
    parser.add_argument("--thinking_token_threshold", type=int, default=6400)
    parser.add_argument("--memory_token_threshold", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main_groups(parse_args())
