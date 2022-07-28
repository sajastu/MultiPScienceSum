#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

USER=$1
export MODEL_NAME=allenai/led-large-16384-arxiv
#export MODEL_NAME=allenai/led-base-16384
export DS_DIR=/disk1/sajad/datasets/sci/mup/hf_format/
export HF_DATASETS_CACHE=/disk0/$USER/.cache/huggingface

#### Model specific args

export SINGLE_WR_WRITE="/disk1/sajad/datasets/sci/mup/single_tokenized/"
export TOPIC_FILE_PATH="/disk1/sajad/datasets/sci/mup/bert_data/"



CUDA_VISIBLE_DEVICES=1 python run_summarization.py \
    --mode train \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --output_dir /disk0/$USER/.cache/sci-trained-models/mup-led-arxiv-4096-6144-AllSents-PrepConc \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --num_train_epochs 5 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --text_column source \
    --summary_column summary \
    --overwrite_output_dir \
    --evaluation_strategy steps --warmup_ratio 0.05 --logging_steps 100 \
    --predict_with_generate \
    --max_grad_norm 1 \
    --lr_scheduler_type linear \
    --eval_steps 4180 --save_steps 4180 \
    --train_file $DS_DIR/train.parquet \
    --validation_file $DS_DIR/val.parquet \
    --do_train \
    --do_eval \
    --report_to wandb \
    --run_name mup-led-arxiv-4096-6144-AllSents-PrepConc \
    --max_source_length 6144 \
    --preprocessing_num_workers 4 \
    --metric_for_best_model rougeL_f \
    --greater_is_better True \
    --topic_file_path $TOPIC_FILE_PATH \

    #    --test_file $DS_DIR/test.reduced.complete.parquet \
#    --do_predict \
#    --filtered_ids "7cbbcd36c5af118c7caad20f1b2cf159"
#    --filtered_ids "183a64018088087429e503d3f533ea89"

