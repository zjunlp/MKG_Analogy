for lr in 1e-2
do
echo ${lr}
python learn.py \
    --dataset="analogy" \
    --model="ComplEx" \
    --batch_size=1000 \
    --learning_rate=${lr} \
    --max_epochs=300
done