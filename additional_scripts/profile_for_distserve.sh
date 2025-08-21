#!/bin/bash

profile_model() {
    local MODEL="$1"
    local OUTPUT_DIR="$2"
    local TP_SIZE="$3"

    BATCH_SIZE=(1 2 4 8 16 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512)
    INPUT_LENS=(4 8 16 32 48 64 96 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 2176 2304 2432 2560 2688 2816 2944 3072)
    OUTPUT_LENS=(16)

    for INPUT in "${INPUT_LENS[@]}"; do
        OUTPUT_FILE="$OUTPUT_DIR/profile_input_${INPUT}_output_${OUTPUT_LENS}.log"
        python -m sglang.bench_one_batch \
            --model-path "$MODEL" \
            --tp-size "$TP_SIZE" \
            --disable-radix-cache \
            --batch "${BATCH_SIZE[@]}" \
            --input-len "$INPUT" \
            --output-len "${OUTPUT_LENS[@]}" \
            --run-name "TP${TP_SIZE}" \
            --log-decode-step 2 \
            | tee "$OUTPUT_FILE"
    done
}

MODEL="meta-llama/Llama-3.1-8B-Instruct"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

profile_model "$MODEL" "$OUTPUT_DIR" 1
profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4
profile_model "$MODEL" "$OUTPUT_DIR" 8


################################################

MODEL="Qwen/Qwen2.5-14B-Instruct"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

profile_model "$MODEL" "$OUTPUT_DIR" 1
profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4
profile_model "$MODEL" "$OUTPUT_DIR" 8

################################################

MODEL="microsoft/phi-4"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

profile_model "$MODEL" "$OUTPUT_DIR" 1
profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4
profile_model "$MODEL" "$OUTPUT_DIR" 8

################################################

MODEL="Qwen/Qwen3-30B-A3B"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

profile_model "$MODEL" "$OUTPUT_DIR" 1
profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4

################################################

MODEL="google/gemma-2-27b-it"
SHORT_MODEL="${MODEL#*/}"
OUTPUT_DIR=Distserve_profile_$SHORT_MODEL
mkdir -p "$OUTPUT_DIR"

profile_model "$MODEL" "$OUTPUT_DIR" 1
profile_model "$MODEL" "$OUTPUT_DIR" 2
profile_model "$MODEL" "$OUTPUT_DIR" 4
