#!/bin/bash

# Parameter arrays for image-only CNN
dense_sizes=(128 512)
dense_k_reg=(0.05)
learning_rates=(0.0005)
freeze_base_options=(true)

# Fixed parameters
batch_size=128
epochs=100
dataset_perc=0.5
img_size=224
use_mobilenet_preprocessing=true

# Counter for tracking progress
total_jobs=$((${#dense_sizes[@]} * ${#learning_rates[@]} * ${#dense_k_reg[@]} * ${#freeze_base_options[@]}))
current_job=0

echo "Starting image-only CNN parameter grid search with $total_jobs total combinations..."
echo "Parameters:"
echo "  dense_sizes: ${dense_sizes[*]}"
echo "  learning_rates: ${learning_rates[*]}"
echo "  dense_k_reg: ${dense_k_reg[*]}"
echo "  freeze_base_options: ${freeze_base_options[*]}"
echo "Fixed parameters:"
echo "  batch_size: $batch_size"
echo "  epochs: $epochs"
echo "  img_size: $img_size"
echo "  dataset_perc: $dataset_perc"
echo "----------------------------------------"

# Nested loops for all parameter combinations
for d_size in "${dense_sizes[@]}"; do
    for l_rate in "${learning_rates[@]}"; do
        for dense_k in "${dense_k_reg[@]}"; do
            for freeze_base in "${freeze_base_options[@]}"; do
                current_job=$((current_job + 1))                
                echo "[$current_job/$total_jobs] Running with:"
                echo "  dense_size=$d_size, learning_rate=$l_rate"
                echo "  dense_k_reg=$dense_k, freeze_base=$freeze_base"
                
                # Run the command
                nice -n 10 python MobNetV3-imageonly.py \
                    --epochs $epochs \
                    --batch_size $batch_size \
                    --dense_size $d_size \
                    --dataset_perc $dataset_perc \
                    --l_rate $l_rate \
                    --dense_k_reg $dense_k \
                    --img_size $img_size \
                    --freeze_base $freeze_base \
                    --use_mobilenet_preprocessing $use_mobilenet_preprocessing
                
                echo "Job $current_job completed."
                echo "----------------------------------------"
            done
        done
    done
done

echo "All parameter combinations completed!"
echo "Grid search completed successfully!"
echo "Results saved in the specified output directory"