#!/bin/bash

# Experiment
PROJECT_NAME='mma'
EXP_NAME='MB_GSPO-DS_DAPO-Math_Qwen3-8B-Base'
EPOCHS=8
CKPTS_DIR='volume'
ROLLOUT_DATA_DIR=null
CKPT_FREQUENCY=20
## Validation
VAL_BEFORE_TRAIN=False
DENSE_VAL_FREQUENCY=10 #dense
SPARSE_VAL_FREQUENCY=50
DENSE_CUTOFF=250
VAL_N=16
VAL_GENERATIONS=10




# Data
TRAIN_FILE='data/dapo-math-17k.parquet'
TEST_FILE='data/math_eval.parquet'
PROMPT_KEY='prompt' #using processed (deduplicated) DAPO-Math-17k dataset
REWARD_MANAGER='dapo'
CUSTOM_REWARD_FUNC_PATH='bmc_reward.py'
CUSTOM_REWARD_FUNC_NAME='reward_func'


# Model
MODEL_PATH='Qwen/Qwen3-8B-Base'
## Model Sampling Params (Temperature, Top-P, Top-K)
TRAIN_TEMPERATURE=1.0 #avoid entropy compression
TRAIN_TOP_P=1.0 #avoid entropy compression
TRAIN_TOP_K=-1 # 0 for HF rollout, -1 for vLLM rollout
VAL_TEMPERATURE=0.6 
VAL_TOP_P=0.95 
VAL_TOP_K=-1 # 0 for HF rollout, -1 for vLLM rollout
## LoRA
LORA_RANK=0 #if 0, LoRA won't be used
LORA_ALPHA=0 # N/A if LORA_RANK==0

# Critic
CRITIC_WARMUP=0 #no critic initialization

# Algorithm
## General features
POLICY_LOSS='gspo'
LOSS_AGG_MODE='seq-mean-token-mean' 
ADV_ESTIMATOR='grpo'
NORM_ADV_BY_STD=True
ACTOR_LEARNING_RATE=1e-6
LEARNING_RATE_WARMUP_STEPS=0 #Prolonged-RL, DeepScaler, and GRESO kept it constant--so we will, too
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
ENTROPY_COEFF=0.0 #original DAPO script featued zero entropy coef (specifically: run_dapo_qwen2.5_32b.sh)

## GSPO feature: Clip Ratios
CLIP_RATIO_HIGH=4e-4 #from gspo_trainer/run_qwen30b_gspo.sh
CLIP_RATIO_LOW=3e-4 #from gspo_trainer/run_qwen30b_gspo.sh


## GSPO/DAPO feature: No KL
USE_KL_LOSS=False
USE_KL_IN_REWARD=False 
KL_COEF=0.0 
KL_LOSS_COEF=0.0


## DAPO feature: Dynamic Sampling
FILTER_GROUPS=True # IMPORTANT; ENABLES DYNAMIC SAMPLING
FILTER_METRIC=acc
MAX_DYN_BATCHES=0

## DAPO feature: Overlong Reward Shaping (NOT USED FOR MB)
OVERLONG_BUFFER=False
OVERLONG_BUFFER_LENGTH=410 #10% of max length; try to allow for longer responses
OVERLONG_PENALTY=1.0
OVERLONG_LOG=False


# Batch Sizes, Rollouts, and Context Lengths
TRAIN_BATCH_SIZE=128
GEN_BATCH_SIZE=256
MINI_BATCH_SIZE=32
BALANCE_BATCH=True
DYNAMIC_BATCH_SIZE=True
ROLLOUT_TYPE=vllm
K_ROLLOUTS=8
MAX_PROMPT_LENGTH=3000 #$((1024 * 2)) #need to handle evaluations and training set, but for DAPO-Math, the max prompt length is ~1600
MAX_RESPONSE_LENGTH=$((1024 * 4))
MAX_TOKEN_TOTAL=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) #from verl's Performance Tuning guide
MAX_TOKEN_LEN_PER_GPU=$((MAX_TOKEN_TOTAL * 2))
INFER_MAX_TOKEN_LEN_PER_GPU=$((MAX_TOKEN_TOTAL * 3))


# Compute
NNODES=1
NGPUS_PER_NODE=8
NCPU_CORES=32
STRATEGY=fsdp2
FSDP_SIZE=-1 #auto
ROLLOUT_GPU_UTILIZATION=0.85
SP_SIZE=1 
GEN_TP=1 
OFFLOAD=True
GRADIENT_CHECKPOINTING=True
ENTROPY_CHECKPOINTING=True
CHUNKED_PREFILL=False

# Run
python -m main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key="${PROMPT_KEY}" \
    data.truncation='left' \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNC_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNC_NAME} \
    actor_rollout_ref.rollout.n=${K_ROLLOUTS} \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD} \
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    algorithm.filter_groups.enable=${FILTER_GROUPS} \
    algorithm.filter_groups.metric=${FILTER_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${MAX_DYN_BATCHES} \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${DYNAMIC_BATCH_SIZE} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${DYNAMIC_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${DYNAMIC_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=${LORA_RANK} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
    actor_rollout_ref.actor.entropy_checkpointing=${ENTROPY_CHECKPOINTING} \
    actor_rollout_ref.ref.entropy_checkpointing=${ENTROPY_CHECKPOINTING} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LEARNING_RATE} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LEARNING_RATE_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.rollout.name=${ROLLOUT_TYPE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTILIZATION} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${CHUNKED_PREFILL} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKEN_TOTAL} \
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P} \
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${OFFLOAD} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE} \
    actor_rollout_ref.actor.strategy=${STRATEGY} \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.overlong_buffer.enable=${OVERLONG_BUFFER} \
    reward_model.overlong_buffer.len=${OVERLONG_BUFFER_LENGTH} \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_PENALTY} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LENGTH} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=${OVERLONG_LOG} \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.balance_batch=${BALANCE_BATCH} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    +trainer.n_cpu_cores=${NCPU_CORES} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.test_freq=${DENSE_VAL_FREQUENCY} \
    +trainer.sparse_test_freq=${SPARSE_VAL_FREQUENCY} \
    +trainer.dense_cutoff=${DENSE_CUTOFF} \
    trainer.save_freq=${CKPT_FREQUENCY} \
    trainer.total_epochs=${EPOCHS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=${VAL_GENERATIONS}