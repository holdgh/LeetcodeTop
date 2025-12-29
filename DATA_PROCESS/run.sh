set -e
set -u


export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1 \


SCRIPT_DIR=$(cd $(dirname $0); pwd)
WORK_DIR=$SCRIPT_DIR/../..
CONFIG_DIR=$WORK_DIR/scripts/config
MODEL_DIR=$WORK_DIR/models
DATA_DIR=$WORK_DIR/data

torchrun --nnodes 1 --nproc_per_node 1 $WORK_DIR/sft.py \
    --base_model /home/models/InternVL3-1B \
    --output_dir $MODEL_DIR/InternVL3-1B \
    --use_rasa \
    --lora_r 8 \
    --rasa_k 1 \
    --lora_alpha 32 \
    --chat_template_name internvl3-chat \
    --data_name normal.json \
    --data_dir /nas_data/上飞/实验数据 \
    --batch_size 192 \
    --micro_batch_size 2 \
    --num_train_epochs 8 \
    --use_lion \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --save_strategy epoch \
    --bf16 \
    --gc \
    --group_by_length \
    --deepspeed $CONFIG_DIR/deepspeed/deepspeed_config_zero2_lion.json

