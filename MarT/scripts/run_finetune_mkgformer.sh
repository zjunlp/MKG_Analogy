for lr in 5e-5
do
for alpha in 0.43
do
echo $lr
echo $alpha
python main.py \
   --gpus "0," \
   --max_epochs=15  \
   --num_workers=4 \
   --model_name_or_path bert-base-uncased \
   --visual_model_path openai/clip-vit-base-patch32 \
   --accumulate_grad_batches 1 \
   --model_class MKGformerKGC \
   --batch_size 32 \
   --pretrain 0 \
   --bce 0 \
   --check_val_every_n_epoch 1 \
   --overwrite_cache \
   --data_dir dataset/MARS \
   --pretrain_path dataset/MarKG \
   --eval_batch_size 128 \
   --max_seq_length 128 \
   --lr $lr \
   --alpha $alpha \
   --checkpoint pre_train_checkpoint_of_mkgformer
done
done