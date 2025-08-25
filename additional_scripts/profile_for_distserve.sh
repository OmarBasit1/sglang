#!/bin/bash

profile_model() {
    local MODEL="$1"
    local OUTPUT_DIR="$2"
    local TP_SIZE="$3"

    # BATCH_SIZE=(1 2 4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512)
    # INPUT_LENS=(4 8 16 32 48 64 96 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072)
    OUTPUT_LENS=(8)

    for INPUT in "${INPUT_LENS[@]}"; do
        OUTPUT_FILE="$OUTPUT_DIR/profile_input_${INPUT}_output_${OUTPUT_LENS}_TP_${TP_SIZE}.log"
        python -m sglang.bench_one_batch_for_distserve \
            --model-path "$MODEL" \
            --tp-size "$TP_SIZE" \
            --disable-radix-cache \
            --batch "${BATCH_SIZE[@]}" \
            --input-len "$INPUT" \
            --output-len "${OUTPUT_LENS[@]}" \
            --run-name "TP${TP_SIZE}" \
            --log-decode-step 1 \
            --repeat-prefill 4 \
            | tee "$OUTPUT_FILE"
    done
}

MODEL="meta-llama/Llama-3.1-8B-Instruct"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=./new_distserve_logs/Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

BATCH_SIZE=(1 2 4 8 16)
INPUT_LENS=(2176 2304 2432 2560 2688 2816 2944 3072)
# # CUDA_VISIBLE_DEVICES=0 profile_model "$MODEL" "$OUTPUT_DIR" 1
# # CUDA_VISIBLE_DEVICES=0,1 profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4



################################################

MODEL="Qwen/Qwen2.5-14B-Instruct"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=./new_distserve_logs/Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

BATCH_SIZE=(1 2 4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512)
INPUT_LENS=(4 8 16 32 48 64 96)
# # CUDA_VISIBLE_DEVICES=1 profile_model "$MODEL" "$OUTPUT_DIR" 1
# CUDA_VISIBLE_DEVICES=2,3 profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4

BATCH_SIZE=(1 2 4 8 16 32 64 96 128 160 192)
INPUT_LENS=(128 256 384 512)
profile_model "$MODEL" "$OUTPUT_DIR" 4

BATCH_SIZE=(1 2 4 8 16 32 64)
INPUT_LENS=(640 768 896 1024 1152 1280 1408 1536 1664 )
profile_model "$MODEL" "$OUTPUT_DIR" 4

BATCH_SIZE=(1 2 4 8 16)
INPUT_LENS=(1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072)
profile_model "$MODEL" "$OUTPUT_DIR" 4

# ################################################

# MODEL="microsoft/phi-4"
# SHORT_MODEL="${MODEL#*/}"
# OUTPUT_DIR=./new_distserve_logs/Distserve_profile_$SHORT_MODEL
# mkdir -p "$OUTPUT_DIR"

# BATCH_SIZE=(1 2 4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512)
# INPUT_LENS=(4 8 16 32 48 64 96 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072)
# CUDA_VISIBLE_DEVICES=2 profile_model "$MODEL" "$OUTPUT_DIR" 1
# CUDA_VISIBLE_DEVICES=2,3 profile_model "$MODEL" "$OUTPUT_DIR" 2


# # # ################################################

# MODEL="google/gemma-2-27b-it"
# SHORT_MODEL="${MODEL#*/}"
# OUTPUT_DIR=./new_distserve_logs/Distserve_profile_$SHORT_MODEL
# mkdir -p "$OUTPUT_DIR"

# BATCH_SIZE=(1 2 4 8 16 32 64 96)
# INPUT_LENS=(384 512)
# # profile_model "$MODEL" "$OUTPUT_DIR" 2
# profile_model "$MODEL" "$OUTPUT_DIR" 4


