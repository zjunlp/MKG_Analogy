for lr in 5e-3
do
echo ${lr}
python learn.py \
    --dataset="analogy" \
    --model="Analogy" \
    --batch_size=1000 \
    --learning_rate=${lr} \
    --max_epochs=300 \
    --finetune \
    --ckpt="your_ckpt_path"
done