set -eu

exp_dir="work_dir/s3dis-pointnet-v2"
rm -rf $exp_dir

cfg_file=configs/s3dis/pointnet-v2.py

NUM_GPUS=1
NUM_NODES=1
MASTER_PORT=29500

export PYTHONPATH=$(pwd)


python scripts/train.py \
    --config $cfg_file \
    --exp-dir $exp_dir \
    --batch-size 2 \
    --num-workers 2 \
    --lr 3e-4 \
    --epochs 50 \
    --precision 32 \
    --auto-resume true \
    --mode train \
    --tb-logs-dir $exp_dir/tb_logs \
    --limit-batches 20

