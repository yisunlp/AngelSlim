export PYTORCH_ALLOC_CONF="expandable_segments:True"
torchrun --nproc_per_node=8 tools/run.py -c "configs/qwen3/qat/fp8_static/learn_scale/qwen3-30b-a3b_fp8_static_end2end_learn_scale_lwc_zero3.yaml" 


# source /apdcephfs_zwfy2/share_301053287/brunosu/init_scripts/init_conda.sh

# echo $NODE_IP_LIST > env.txt 2>&1
# sed "s/:/ slots=/g" env.txt | sed "s/,/\n/g" >  "hostfile"
# MASTER_IP=$(head -1 hostfile | awk '{print $1}')

# deepspeed --hostfile=hostfile \
#     --master_addr=$MASTER_IP \
#     --master_port=29500 \
#     tools/run.py \
#     -c "hunyuan.yaml" \
#     --lm-eval \
#     --ppl-eval