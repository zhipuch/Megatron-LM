# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import os
import warnings
from copy import deepcopy

import torch
from glm4v_config import get_language_model_config, get_vision_model_config, get_vision_projection_config
from layer_specs import get_layer_spec, get_layer_spec_te, get_mlp_module_spec, get_norm_mlp_module_spec_te, get_proj_module_spec_te,  get_vit_layer_with_transformer_engine_spec_for_eva_clip

from megatron.core.models.multimodal.glm4v_model import GLM4V
from megatron.core.models.vision.eva_clip_model import get_num_image_embeddings
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu

def get_checkpoint_name(checkpoints_path,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None):
    """Determine the directory name for this rank's checkpoint."""


    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, 
                            f'mp_rank_{tensor_rank:02d}_glm4v.pt')
    else:
        common_path = os.path.join(checkpoints_path, 
                        f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}_glm4v.pt')

    return common_path

def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True, parallel_output=True
) -> GLM4V:
    """Builds the model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A multimodal model.
    """
    args = get_args()

    use_te = args.use_te

    print_rank_0('building a multimodal model ...')

    num_image_embeddings = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, args.vision_model_type,
        args.disable_vision_class_token, 1
    )
    old_seq_length = args.seq_length
    args.seq_length = args.encoder_seq_length = num_image_embeddings
    #if torch.distributed.get_rank() == 0 and old_seq_length != args.seq_length:
    #    warnings.warn(
    #        f"Changed seq_length and encoder_seq_length (vision model sequence length) from {old_seq_length} to num_image_tokens ({num_image_embeddings})"
    #    )

    max_num_image_embeddings = (args.max_num_tiles + int(args.use_thumbnail)) * num_image_embeddings

    assert (
        args.decoder_seq_length is not None
    ), "Please provide --decoder-seq-length to set the language model sequence length"
    assert (
        args.decoder_seq_length > max_num_image_embeddings
    ), "Language model sequence length must be greater than the maximum number of image embeddings"
    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn(
            f"Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the maximum language model sequence length"
        )

    base_config = core_transformer_config_from_args(get_args())
    base_config.language_model_type = args.language_model_type
    base_config.vision_model_type = args.vision_model_type
    base_config.calculate_per_token_loss = True

    language_config = deepcopy(base_config)
    language_config = get_language_model_config(language_config)

    if use_te:
        language_transformer_layer_spec = get_layer_spec_te(
            is_vit=False
        )  # TENorm detects LayerNorm/RMS automatically.
    else:
        language_transformer_layer_spec = get_layer_spec(
            is_vit=False, normalization=language_config.normalization
        )

    vision_config = deepcopy(base_config)
    vision_config = get_vision_model_config(
        vision_config, apply_query_key_layer_scaling=args.apply_query_key_layer_scaling
    )

    vision_model_type = args.vision_model_type
    if vision_model_type in ["clip", "siglip"]:
        if use_te:
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec_for_eva_clip()  # TENorm detects LayerNorm/RMS automatically.
        else:
            vision_transformer_layer_spec = get_layer_spec(
                is_vit=True, normalization=vision_config.normalization
            )
    else:
        raise RuntimeError("unsupported vision model type", vision_model_type)
    
    vision_projection_config = deepcopy(base_config)
    vision_projection_config = get_vision_projection_config(
        vision_projection_config, language_config.hidden_size
    )

    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "vision model and projection can only live on 1 pipeline stage."
        vision_config.pipeline_model_parallel_size = args.encoder_pipeline_model_parallel_size
        vision_projection_config.pipeline_model_parallel_size = (
            args.encoder_pipeline_model_parallel_size
        )
        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size
            vision_projection_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    vision_projection_modules = get_proj_module_spec_te()

    model = GLM4V(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=151552,
        language_max_sequence_length=args.decoder_seq_length,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_modules,
        vision_projection_type="mlp",
        allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
    )

    model.freeze(
        freeze_language_model=args.freeze_LM,
        freeze_vision_model=args.freeze_ViT,
        freeze_vision_projection=False,
    )
    ckpt_path = get_checkpoint_name(args.save)

    state_dict = torch.load(ckpt_path, map_location=f"cuda:{torch.cuda.current_device()}")
    model.load_state_dict(state_dict, False)    

    return model