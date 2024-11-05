#/bin/bash
MCORE_MISTRAL='/mnt/ceph/develop/yuxuan/opensource-team/chenhao/Megatron-LM/examples/multimodal/abc/lm'   # <path_to_mcore_mistral_model_folder>
MCORE_CLIP='/mnt/ceph/develop/yuxuan/opensource-team/chenhao/Megatron-LM/examples/multimodal/abc/vm'   # <path_to_mcore_clip_model_folder>
OUTPUT_DIR='/mnt/ceph/develop/yuxuan/opensource-team/chenhao/Megatron-LM/examples/multimodal/abc'   # <path_to_output_folder_for_combined_checkpoint>

python examples/multimodal/combine_state_dicts.py \
    --input \
    ${MCORE_MISTRAL}/lmp_rank_00.pt \
    ${MCORE_CLIP}/vmp_rank_00.pt \
    ${MCORE_MISTRAL}/lmp_rank_01.pt \
    ${MCORE_CLIP}/vmp_rank_01.pt \
    --prefixes language_model vision_model language_model vision_model \
    --output \
    ${OUTPUT_DIR}/mp_rank_00_glm4v.pt \
    ${OUTPUT_DIR}/mp_rank_01_glm4v.pt \

echo 1 > ${OUTPUT_DIR}/latest_checkpointed_iteration.txt
