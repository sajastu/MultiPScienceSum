#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
#export HF_DATASETS_CACHE=/disk0/shabnam/.cache/huggingface
#mkdir -p $HF_DATASETS_CACHE
#export MODEL_NAME=/disk1/sajad/sci-trained-models/bart/mentsum/checkpoint-50000/
#export MODEL_NAME=facebook/bart-large-cnn
#export MODEL_NAME=/disk1/sajad/sci-trained-models/grease/cnndm-greaseBart/checkpoint-239150/

#python3 -m torch.distributed.launch --nproc_per_node=2 examples/pytorch/summarization/run_summarization.py \
#CUDA_VISIBLE_DEVICES=0 python examples/pytorch/summarization/run_summarization.py \
#    --model_name_or_path $MODEL_NAME \
#    --output_dir /disk1/sajad/sci-trained-models/bart/mentsum/checkpoint-50000/ \
#    --per_device_train_batch_size=2 \
#    --per_device_eval_batch_size=1 \
#    --learning_rate 3e-5 \
#    --weight_decay 0.01 \
#    --adam_beta2 0.999 \
#    --num_train_epochs 10 \
#    --save_total_limit 5 \
#    --text_column src \
#    --summary_column tldr \
#    --overwrite_output_dir \
#    --evaluation_strategy steps --warmup_steps 2000 --logging_steps 100 \
#    --predict_with_generate \
#    --max_grad_norm 0.1 \
#    --eval_steps 2700 --save_steps 2700 \
#    --train_file $DS_DIR/train.parquet \
#    --validation_file $DS_DIR/val.parquet \
#    --test_file $DS_DIR/test.parquet \
#    --use_gnn True \
#    --do_predict \
#    --filtered_ids "8a3d53d2df7a72d55a45d017a8692996"
#    --load_best_model_at_end True \
#    --greater_is_better True\
#    --metric_for_best_model rougeL \
#    --report_to wandb \
#    --run_name MS-grease-encoder10-node500 \
#    --do_train \
#    --do_eval \

#    --load_best_model_at_end \


#export USER=$1
#export HF_DATASETS_CACHE=/disk0/$USER/.cache/huggingface
#mkdir -p $HF_DATASETS_CACHE

#export MODEL_NAME=/disk0/$USER/.cache/sci-trained-models/grease/mentsum-grease-encoder6-node1000-withAttn-secnd/checkpoint-59400/
#export MODEL_NAME=/disk1/sajad/sci-trained-models/bart/mentsum/checkpoint-50000/
#export MODEL_NAME=/disk0/$USER/.cache/sci-trained-models/grease/mentsum-grease-encoder6-node750-withAttn-v5/
#export MODEL_NAME=/disk0/sajad/.cache/sci-trained-models/grease/mentsum-grease-encoder6-node1000-withAttn-third/checkpoint-10/
USER=$1
export MODEL_NAME=allenai/led-large-16384-arxiv
#export MODEL_NAME=allenai/led-base-16384
export DS_DIR=/disk1/sajad/datasets/sci/mup/hf_format/
#python3 -m torch.distributed.launch --nproc_per_node=2 run_summarization.py \
CUDA_VISIBLE_DEVICES=1 python run_summarization.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir /disk0/$USER/.cache/sci-trained-models/mup-led-arxiv-decAttn-4 \
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
    --eval_steps 15 --save_steps 4150 \
    --train_file $DS_DIR/train.parquet \
    --validation_file $DS_DIR/val.parquet \
    --do_train \
    --do_eval \
    --report_to none \
    --run_name mup-led-arxiv-decAttn-4

#    --metric_for_best_model rougeL \
    #    --test_file $DS_DIR/test.reduced.complete.parquet \
#    --do_predict \
#    --filtered_ids "7cbbcd36c5af118c7caad20f1b2cf159"
#    --filtered_ids "183a64018088087429e503d3f533ea89"

