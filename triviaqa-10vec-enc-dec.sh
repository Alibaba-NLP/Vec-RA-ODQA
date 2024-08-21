export TRANSFORMERS_CACHE=~/.cache/huggingface/
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
export WANDB_DISABLED=true

# You might want to change it to [huggingface repo] or [your local path].
MODEL_NAME=~/models/bloomz-1b7;

OUTPUT_DIR="output/triviaqa_bloomz1b_5k_10vec_enc_dec";
mkdir -p $OUTPUT_DIR;

TORCH_USE_CUDA_DSA=1 \
CUDA_LAUNCH_BLOCKING=1 \
python -u \
    ./src/run_clm_enc_dec.py \
    --model_name_or_path $MODEL_NAME \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --fp16 True \
    --fp16_full_eval True \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --train_file ./data/triviaqa_train_fullk_rest_online_template.txt.json  \
    --validation_file ./data/triviaqa_dev_fullk_rest_online_template.txt.json  \
    --prompt_max_length 2048 \
    --answer_max_length 512 \
    --num_train_epochs 3 \
    --save_strategy steps \
    --save_steps 6000 \
    --save_total_limit 10 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --eval_steps 6000 \
    --logging_steps 6000 \
    --gradient_accumulation_steps 4 \
    --eval_accumulation_steps 4 \
    --use_vector_num 10 \
    --preprocessing_num_workers 8 \
    --train_encoder True \
    --train_decoder True \
    --report_to none