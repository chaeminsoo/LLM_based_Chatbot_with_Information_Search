accelerate launch train_instruct_tuned_model.py\
    --model_name_or_path "EleutherAI/polyglot-ko-12.8b" \
    --dataset_path "{데이터 경로}" \
    --max_steps 1000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1\
    --eval_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --optim "paged_adamw_8bit"\
    --max_length 256 \
    --r 8 \
    --lora_dropout 0.05 \
    --lora_alpha 32 \
    --evaluation_strategy "steps" \
    --eval_step 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --save_strategy "epoch" \
    --max_eval_sample 100 \
    --lr_scheduler_type "linear" \
    --bf16 true \
    --warmup_ratio 0.06 \
    --do_train \
    --output_dir "{저당 경로}/KoRani-12.8b" \
    --save_total_limit 1 \
    --load_best_model_at_end