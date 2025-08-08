#!/bin/bash

# Parameter arrays
vocab_limits=(25000 35000)
dense_sizes=(64 128)
# embedding_dims=(250)
conv_filters=(128)

# Fixed parameters
epochs=100
max_seq_len=50
dataset_perc=1
l_rate=0.001
embedding_dim=250

# Counter for tracking progress
total_jobs=$((${#vocab_limits[@]} * ${#dense_sizes[@]} * ${#conv_filters[@]}))
current_job=0

echo "Starting parameter grid search with $total_jobs total combinations..."
echo "Parameters:"
echo "  vocab_limits: ${vocab_limits[*]}"
echo "  dense_sizes: ${dense_sizes[*]}"
#echo "  embedding_dims: ${embedding_dims[*]}"
echo "  conv_filters: ${conv_filters[*]}"
echo "----------------------------------------"

# Nested loops for all parameter combinations
for vocab_limit in "${vocab_limits[@]}"; do
    for d_size in "${dense_sizes[@]}"; do
        for conv_filter in "${conv_filters[@]}"; do
            current_job=$((current_job + 1))                
            echo "[$current_job/$total_jobs] Running with:"
            echo "  vocab_limit=$vocab_limit, dense_size1=$d_size, embedding_dim=$embedding_dim"
            echo "  conv_filters=$conv_filter"
                
            # Run the command
            nice -n 10 python cnn-multiscale.py \
                --epochs $epochs \
                --vocab_limit $vocab_limit \
                --emb_dim $embedding_dim \
                --max_seq_len $max_seq_len \
                --dataset_perc $dataset_perc \
                --l_rate $l_rate \
                --conv_filters $conv_filter
            echo "Job $current_job completed."
            echo "----------------------------------------"
        done
    done
done


echo "All parameter combinations completed!"
