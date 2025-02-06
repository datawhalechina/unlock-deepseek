# 如果你要限制计算卡编号，请在这里设置，例如只使用 cuda:1-3，如果不用限制，就删除下面这行
export CUDA_VISIBLE_DEVICES=1,2,3

accelerate launch \
    --num_processes 2 \
    --config_file deepspeed_zero3.yaml \
    train_Datawhale-R1.py \
    --config Datawhale-R1.yaml