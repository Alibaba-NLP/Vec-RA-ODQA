python3 -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 3 \
  --num_prompt_instructions 1 \
  --model_name="text-davinci-003" \
  --seed_tasks_path="seed_tasks.min.jsonl"