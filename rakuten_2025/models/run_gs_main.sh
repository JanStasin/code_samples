#!/bin/bash

# Parameter arrays
vocab_limits=(25000)
dense_sizes=(64)
conv_k_reg=(0.05)
dense_k_reg=(0.075)

# Fixed parameters
batch_size=128
epochs=100
l_rate=0.0001
dataset_perc=1.
conv_filters=128
embedding_dims=250

# Counter for tracking progress
total_jobs=$((${#vocab_limits[@]} * ${#learning_rates[@]} * ${#conv_k_reg[@]} * ${#dense_k_reg[@]}))
current_job=0

echo "Starting parameter grid search with $total_jobs total combinations..."
echo "Parameters:"
echo "  vocab_limits: ${vocab_limits[*]}"
echo "  dense_sizes: ${dense_sizes[*]}"
echo "  conv_k_reg: ${conv_k_reg[*]}"
echo "  dense_k_reg: ${dense_k_reg[*]}"
echo "  embedding_dims: ${embedding_dims[*]}"
echo "----------------------------------------"

# Nested loops for all parameter combinations
for vocab_limit in "${vocab_limits[@]}"; do
    for d_size in "${dense_sizes[@]}"; do
        for conv_k in "${conv_k_reg[@]}"; do
            for dense_k in "${dense_k_reg[@]}"; do
                current_job=$((current_job + 1))                
                echo "[$current_job/$total_jobs] Running with:"
                echo "  vocab_limit=$vocab_limit, dense_size1=$d_size ,learning_rate=$l_rate"
                echo "  conv_k_reg=$conv_k, dense_k_reg=$dense_k"
                
                # Run the command
                nice -n 10 python MobNetV3-multimodal.py \
                    --epochs $epochs \
                    --batch_size $batch_size \
                    --dense_size $d_size \
                    --vocab_limit $vocab_limit \
                    --emb_dim $embedding_dims \
                    --dataset_perc $dataset_perc \
                    --l_rate $l_rate \
                    --conv_filters $conv_filters \
                    --conv_k_reg $conv_k \
                    --dense_k_reg $dense_k
                echo "Job $current_job completed."
                echo "----------------------------------------"
            done
        done
    done

echo "All parameter combinations completed!"
done
echo "Grid search completed successfully!"
echo "Results saved in the specified output directory"