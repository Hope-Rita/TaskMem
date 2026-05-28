# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM-judge based reward used during episodic memory training.

The judge looks at each candidate description together with the original video
and the preceding context, then assigns:
- correctness: factual alignment with the visuals and subtitles.
- label:       a redundancy/format filter (no boxes, no meta phrases, etc.).
"""
import os
import re
import json
import base64
import random
from time import sleep
from typing import Dict, Any, List

import openai

USER_PROMPT_EPISODIC_VIDEO = """
You are given a video input and groups of [Description]. Each description must be evaluated independently; the correctness of one description does not affect others.
Your task is to evaluate if each [Description] is factually accurate only based on visuals and subtitles (ignore audio). For spoken content, verify them only based on the displayed subtitles, ignore any audio.
Assign exactly one label:
1: Correct — Everything written in the description must be internally coherent and supported by the video or subtitles.
0: Incorrect — Any mismatches or hallucinations in the description.
Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.
{{
    "correctness_rationale": "Short explanation for marking this description as 1 or 0",
    "correctness": 1 or 0,
}}

Descriptions to verify:
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
    "label": 1 or 0,
}}

[Context]:
{preceding_json}

candidate description to verify:
{blocks_text}

""".strip()

FACE_PATTERN = re.compile(r"\bface[ _]\d+\b", re.IGNORECASE)

TEMPERATURE = 1e-6
MAX_RETRIES = 5

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "api_config.json",
)
_API_CONFIG_PATH = os.environ.get("TASKMEM_API_CONFIG", _DEFAULT_CONFIG_PATH)

with open(_API_CONFIG_PATH, "r", encoding="utf-8") as _f:
    config = json.load(_f)

client = {}
for model_name in config.keys():
    if isinstance(config[model_name], list):
        client[model_name] = [openai.AzureOpenAI(
            azure_endpoint=conf["azure_endpoint"],
            api_version=conf["api_version"],
            api_key=conf["api_key"],
        ) for conf in config[model_name]]
    else:
        client[model_name] = openai.AzureOpenAI(
            azure_endpoint=config[model_name]["azure_endpoint"],
            api_version=config[model_name]["api_version"],
            api_key=config[model_name]["api_key"],
        )


def get_response_with_retry(model, messages, timeout=30):
    for i in range(MAX_RETRIES):
        try:
            if isinstance(client[model], list):
                selected_model = random.choice(client[model])
            else:
                selected_model = client[model]
            response = selected_model.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
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
    messages = [{"role": "system", "content": "You are an expert in video understanding."}]
    content = []
    for input in inputs:
        if input["type"] == "text":
            content.append(input)
        elif input["type"] == "video":
            base64_video = base64.b64encode(open(input["video"], "rb").read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:video/mp4;base64,{base64_video}"},
            })
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages


def _invalid_face_tag(desc: str) -> bool:
    if re.search(r'<face>', desc, re.IGNORECASE):
        return True

    suspicious_pattern = re.compile(r'face[\s_]*\d+', re.IGNORECASE)

    for m in suspicious_pattern.finditer(desc):
        s, e = m.span()
        token = m.group()

        if token != f"face_{token.split('_')[-1]}" or not re.fullmatch(r'face_\d+', token):
            if not re.fullmatch(r'face_\d+', token):
                return True

        has_brackets = (s > 0 and desc[s-1] == '<' and e < len(desc) and desc[e] == '>')
        if not has_brackets:
            return True

    return False


def _extract_json_obj_from_text(responses: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", responses, re.S)
    if not match:
        raise ValueError("No JSON-like object found in response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Missing or invalid JSON object.")
    return parsed


def _parse_llm_response(responses: str, task_type: str) -> Dict[str, Any]:
    parsed = _extract_json_obj_from_text(responses)
    if task_type not in parsed:
        raise ValueError(f"missing '{task_type}' in response")
    return parsed


def _process_task(task_inputs, task_type: str, model_name: str, timeout: int):
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            messages = generate_messages(task_inputs)
            responses = get_response_with_retry(model_name, messages, timeout)
            return _parse_llm_response(responses, task_type)
        except Exception as e:
            last_error = e
            continue
    print(f"[Judge] {task_type} failed: {last_error}")
    return {task_type: 0, f"{task_type}_rationale": ""}


def score_episodic(
    prompt,
    preceding,
    video_path,
    inputs,
    task_type,
    model_name: Dict[str, str],
    timeout: int = 60,
):
    prompt_item = prompt[task_type].format(preceding_json=preceding, blocks_text=inputs)

    judge_inputs = (
        [{"type": "text", "text": prompt_item}]
        if task_type == "label"
        else [{"type": "video", "video": video_path}, {"type": "text", "text": prompt_item}]
    )

    out = _process_task(
        task_inputs=judge_inputs,
        task_type=task_type,
        model_name=model_name[task_type],
        timeout=timeout,
    )
    return out[task_type], out[f"{task_type}_rationale"]


def score_episodic_final(
    context: Any,
    input_memory: str,
    timeout: int = 60,
) -> bool:
    text_items = [it["text"] for it in context if it.get("type") == "text" and "text" in it]
    last_text = text_items[-1] if text_items else ""
    preceding = (
        last_text.split("[Description of the preceding part]:")[1]
        .split("\n\n- If [Description of the preceding part] is empty")[0]
        .strip()
    )
    preceding = preceding if isinstance(preceding, str) else ""

    video_path = None
    for item in context:
        if item["type"] == "video":
            video_path = item["video"]
            break
    assert video_path is not None, "video_path is None"

    prompt = {
        "correctness": USER_PROMPT_EPISODIC_VIDEO,
        "label": USER_PROMPT_EPISODIC_TEXT,
    }
    mode_names = {
        "correctness": "gemini-1.5-pro-002",
        "label": "gpt-4o-2024-11-20",
    }

    c, _ = score_episodic(
        prompt=prompt,
        task_type="correctness",
        preceding=preceding,
        inputs=input_memory,
        model_name=mode_names,
        video_path=video_path,
        timeout=timeout,
    )

    l, _ = score_episodic(
        preceding=preceding,
        inputs=input_memory,
        task_type="label",
        prompt=prompt,
        model_name=mode_names,
        video_path=video_path,
        timeout=timeout,
    )

    return c == 1 and l == 1
