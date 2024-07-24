cd /home/limc/workspace/Pro-Prime/
source /home/limc/miniconda3/bin/activate plm
export PYTHONPATH=$PYTHONPATH:$(pwd)

python sft/mutant/finetune_mutant.py \
--log_gradient_norm="False" \
--model_path="AI4Protein/Prime_690M" \
--tokenizer_path="AI4Protein/Prime_690M" \
--fasta_file="example_data/example.fasta" \
--train_file="example_data/train.csv" \
--valid_file="example_data/valid.csv" \
--test_file="example_data/test.csv" \
--train_batch_size="2" \
--eval_batch_size="2" \
--num_workers="16" \
--seed="42" \
--max_epochs="200" \
--accumulate_grad_batches="8" \
--lr="0.0001" \
--adam_beta1="0.9" \
--adam_beta2="0.999" \
--adam_epsilon="1e-7" \
--gradient_clip_value="1.0" \
--gradient_clip_algorithm="norm" \
--precision="32" \
--weight_decay="0.001" \
--scheduler_type="constant" \
--save_model_dir="finetune_checkpoint" \
--save_model_name="mutant_sft" \
--log_steps="10" \
--logger="tensorboard" \
--logger_project="mutant_sft" \
--logger_run_name="mutant_sft" \
--devices=1 \
--nodes=1 \
--accelerator="gpu" \
--strategy="auto" \
--warmup_steps="0" \
--warmup_max_steps="5000000"