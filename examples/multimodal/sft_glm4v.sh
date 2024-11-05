#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NAME="glm4v"
SOURCE="/mnt/ceph/develop/yuxuan/opensource-team/chenhao/Megatron-LM"
PWD="${SOURCE}/examples/multimodal"
SAVE_PATH="${PWD}/abc"
TOKENIZER_PATH="${SOURCE}/megatron/training/tokenizer/glm4_tokenizer"
CHECKPOINT_PATH="${PWD}/ckpt/${NAME}"
TENSORBOARD_PATH="${PWD}/runs/${NAME}"
DATA_TRAIN="${SOURCE}/examples/multimodal/sft_dataset.yaml"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2

TP_SIZE=2
PP_SIZE=1
VIT_LAYERS="63"
VIT_RECOMPUTE_LAYERS="63"
LM_LAYERS="40"
LM_RECOMPUTE_LAYERS="20"


NHIDDEN=4096
FFN_HIDDEN=13696
NLAYERS=40
NHEADS=32

SEQ_LEN=8192

SAVE_INTERVAL=1000

MAX_STEP=10000
WARMUP_STEP=200
TRAIN_SAMPLES=$(( MAX_STEP * GLOBAL_BATCH_SIZE ))
LR_DECAY_SAMPLES=$(( ( MAX_STEP - WARMUP_STEP ) * GLOBAL_BATCH_SIZE ))
LR_WARMUP_SAMPLES=$(( WARMUP_STEP * GLOBAL_BATCH_SIZE ))

script_path="train.py"

OPTIMIZER_ARGS="
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 3e-5 \
    --min-lr 3e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 0.0 \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --initial-loss-scale 65536 \
"

MODEL_ARGS="
    --bf16 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN \
    --seq-length $SEQ_LEN \
    --tokenizer-type MultimodalTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    --tokenizer-prompt-format glm4v \
    --make-vocab-size-divisible-by 256 \
    --group-query-attention \
    --num-query-groups 2 \
    --max-position-embeddings $SEQ_LEN \
    --num-attention-heads $NHEADS \
    --disable-bias-linear \
    --add-qkv-bias \
    --rotary-percent 0.5 \
    --swiglu \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-position-embedding \
    --normalization RMSNorm \
    --language-model-type=glm-9b \
    --use-mcore-models \
    --manual-gc \
    --decoder-seq-length $SEQ_LEN \
"


#删除了vitconfig
VIT_ARGS="
    --freeze-LM \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 1120 \
    --img-w 1120 \
"

# NOTE: here to define the baseline recompute args to be used
TRAINING_ARGS="
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --eval-iters 10 \
    --eval-interval 500 \
"

DATA_ARGS="
    --data-path ${DATA_TRAIN} \
    --dataloader-type external \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
"

OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --timing-log-level 1 \
    --timing-log-option minmax \
    --save-interval $SAVE_INTERVAL \
    --tensorboard-dir $TENSORBOARD_PATH \
    --wandb-exp-name $NAME \
"

#暂时删除data config SCHEDULE_ARGS
gpt_options="
    $MODEL_ARGS \
    $TRAINING_ARGS \
    $OPTIMIZER_ARGS \
    $OUTPUT_ARGS \
    $VIT_ARGS \
    $DATA_ARGS \
    --distributed-timeout-minutes 40 \
    --init-method-std 0.01 \
    --save $SAVE_PATH \
    --load $CHECKPOINT_PATH \
    --no-load-optim \
    --override-opt_param-scheduler \
"

torchrun --nproc_per_node 2 examples/multimodal/train.py ${gpt_options}
