#!/bin/bash

cd ./vidur
chunk_sizes=(512 1024 2048)
TP=(4 8)

for size in "${chunk_sizes[@]}"; do
    for tp in "${TP[@]}"; do
        python -m vidur.main  \
            --replica_config_device a100 \
            --replica_config_network_device a100_dgx \
            --replica_config_model_name facebook/opt-66b \
            --global_scheduler_config_type lor \
            --sarathi_scheduler_config_chunk_size $size \
            --cluster_config_num_replicas 1 \
            --replica_config_tensor_parallel_size $tp \
            --replica_config_num_pipeline_stages 1 \
            --request_generator_config_type trace_replay \
            --trace_request_generator_config_max_tokens 4096 \
            --trace_request_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
            --trace_request_length_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
            --trace_request_interval_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
            --synthetic_request_generator_config_num_requests 512  \
            --replica_scheduler_config_type sarathi  \
            --sarathi_scheduler_config_batch_size_cap 512  \
            --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
            --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
            --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
    done
done

for size in "${chunk_sizes[@]}"; do
    python -m vidur.main  \
        --replica_config_device a100 \
        --replica_config_network_device a100_dgx \
        --replica_config_model_name facebook/opt-66b \
        --global_scheduler_config_type lor \
        --sarathi_scheduler_config_chunk_size $size \
        --replica_config_tensor_parallel_size 4 \
        --cluster_config_num_replicas 2 \
        --replica_config_num_pipeline_stages 1 \
        --request_generator_config_type trace_replay \
        --trace_request_generator_config_max_tokens 4096 \
        --trace_request_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
        --trace_request_length_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
        --trace_request_interval_generator_config_trace_file ./data/processed_traces/sharegpt_rps0.5_processed.csv \
        --synthetic_request_generator_config_num_requests 512  \
        --replica_scheduler_config_type sarathi  \
        --sarathi_scheduler_config_batch_size_cap 512  \
        --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
        --random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
        --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384
done