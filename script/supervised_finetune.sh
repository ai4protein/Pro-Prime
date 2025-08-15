# Train and Inference
python prime_ft.py \
--fasta example.fasta \
--train_csv example.csv \
--valid_csv example.csv \
--test_csv example.csv \
--prediction_csv example.csv \
--output_csv example.pred.csv \
--train_batch_size 2

# Inference
python prime_ft.py \
--skip_train \
--checkpoint YOUR_CHECKPOINT_PATH \
--fasta example.fasta \
--train_csv example.csv \
--valid_csv example.csv \
--test_csv example.csv \
--prediction_csv example.csv \
--output_csv example.pred.csv \
--train_batch_size 2
