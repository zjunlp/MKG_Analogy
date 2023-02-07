
python main.py \
   --gpus "0," \
   --max_epochs=15 \
   --num_workers=4 \
   --model_name_or_path bert-base-uncased \
   --visual_model_path facebook/flava-full \
   --accumulate_grad_batches 1 \
   --model_class FlavaKGC \
   --batch_size 24 \
   --pretrain 1 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/MARS \
   --pretrain_path dataset/MarKG \
   --eval_batch_size 64 \
   --max_seq_length 96 \
   --lr 5e-5
