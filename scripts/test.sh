set -eu

exp_dir="work_dir/s3dis-pointnet-v2-test"
cfg_file=$exp_dir/config.py

# Auto-detect best checkpoint
CKPT_PATH="${exp_dir}/checkpoints/last.ckpt"

export PYTHONPATH=$(pwd)

python scripts/test.py \
    --config $cfg_file \
    --ckpt-path $CKPT_PATH \
    --save-path $exp_dir/test_results
