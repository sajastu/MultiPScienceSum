#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
USER=$1
export MODEL_NAME=allenai/led-large-16384-arxiv
export DS_DIR=/disk1/sajad/datasets/sci/mup/hf_format/
CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir /disk0/$USER/.cache/sci-trained-models/mup-led-TopicAwareEncAttn-Sectional-0.1TL-2ndRun \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 8 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_ratio 0.05 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 1 \
    --lr_scheduler_type linear \
    --eval_steps 4150 --save_steps 4150 \
    --train_file $DS_DIR/train.parquet \
    --validation_file $DS_DIR/val.parquet \
    --do_train \
    --do_eval \
    --report_to none \
    --run_name mup-led-TopicAwareEncAttn-Sectional-0.1TL-2ndRun \
    --metric_for_best_model rougeL_f \
    --greater_is_better True \

    #    --test_file $DS_DIR/test.reduced.complete.parquet \
#    --do_predict \
#    --filtered_ids "7cbbcd36c5af118c7caad20f1b2cf159"
#    --filtered_ids "183a64018088087429e503d3f533ea89"

