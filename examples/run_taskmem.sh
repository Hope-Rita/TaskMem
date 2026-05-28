#!/usr/bin/env bash
#
# Scenario B — TaskMem (with steer adapter).
#
# Runs the full TaskMem inference pipeline. The video / audio preprocessing
# stages are identical to the baseline; the difference is in the final
# --generate_episodic stage, which loads a Phase-Two checkpoint (base
# Qwen3-VL-30B-A3B + steer adapter) through vLLM with the qwen3vllm_ada plugin.
#
# Pipeline (three sequential `src/main.py` invocations on the same OUTPUT_FOLDER):
#   1. --process_audio                : ASR + diarization via Gemini, writes
#                                        <id>_voice.json
#   2. --process_video --process_voice: face detection + speaker-id, renders
#                                        annotated per-clip mp4s
#   3. --generate_episodic            : Qwen3-VL + steer adapter writes the
#                                        task-focused episodic memory
#
# Prerequisites:
#   pip install -e .          # registers the qwen3vllm_ada vLLM plugin
#   export TASKMEM_API_CONFIG=configs/api_config.local.json
#                             # required for gemini-* ASR / voice matching
#
# Required environment variables:
#   VIDEO_PATH       Path to the input .mp4 file.
#   OUTPUT_FOLDER    Working directory for per-clip artefacts and outputs.
#   TASKMEM_CKPT     Path to a Phase-Two checkpoint directory (a
#                     HuggingFace-format folder produced by TaskMem-PhaseTwo,
#                     containing the base Qwen3-VL weights plus the trained
#                     steer adapter).
#
# Optional environment variables:
#   ASR_MODEL        ASR backend (default: gemini-2.5-pro, API).
#   VOICE_MODEL      Voice -> face matcher (default: gemini-2.5-pro, API).
#   TOTAL_DURATION   Max number of seconds to process (default: 180).
#   ADA_TRAIN_LAYERS Comma-separated list of decoder layers on which to
#                     activate the steer adapter. Defaults to "22", which
#                     matches the released Phase-Two checkpoint. Override
#                     only if you trained your own adapter on different
#                     layers.
#   READ_TAG / WRITE_TAG  Memory tags written by the video stage / episodic stage.
set -e

: "${VIDEO_PATH:?Please set VIDEO_PATH to an input .mp4 file}"
: "${OUTPUT_FOLDER:?Please set OUTPUT_FOLDER for the per-clip outputs}"
: "${TASKMEM_CKPT:?Please set TASKMEM_CKPT to a TaskMem Phase-Two checkpoint directory}"
: "${ASR_MODEL:=gemini-2.5-pro}"
: "${VOICE_MODEL:=gemini-2.5-pro}"
: "${TOTAL_DURATION:=180}"
: "${READ_TAG:=raw}"
: "${WRITE_TAG:=taskmem_ep}"
: "${ADA_TRAIN_LAYERS:=22}"

# 1. ASR / diarization (Gemini by default; no GPU). The adapter env vars
#    are intentionally scoped to step 3 only.
python src/main.py \
    --video_path "$VIDEO_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --asr_model "$ASR_MODEL" \
    --write_memory_tag "$READ_TAG" \
    --end_time "$TOTAL_DURATION" \
    --process_audio

# 2. Face detection + speaker matching.
python src/main.py \
    --video_path "$VIDEO_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --voice_model "$VOICE_MODEL" \
    --write_memory_tag "$READ_TAG" \
    --end_time "$TOTAL_DURATION" \
    --process_video --process_voice

# 3. Episodic memory generation with the TaskMem checkpoint.
#    Adapter plugin and Qwen3-VL vLLM-v1 env vars are scoped to this stage to
#    avoid clashing with other vLLM backends in earlier stages.
env \
    VLLM_PLUGINS=qwen3vllm_ada \
    VLLM_MODELS=qwen3vllm_ada \
    VLLM_USE_V1=1 \
    VLLM_ADA_TRAIN_LAYERS="$ADA_TRAIN_LAYERS" \
    python src/main.py \
        --video_path "$VIDEO_PATH" \
        --video_folder "$OUTPUT_FOLDER" \
        --episodic_folder "$OUTPUT_FOLDER" \
        --episodic_model qwen3_vl_vllm \
        --episodic_model_path "$TASKMEM_CKPT" \
        --read_memory_tag "$READ_TAG" \
        --write_memory_tag "$WRITE_TAG" \
        --end_time "$TOTAL_DURATION" \
        --generate_episodic
