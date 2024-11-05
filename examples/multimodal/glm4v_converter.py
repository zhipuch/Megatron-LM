import sys
import os

# 获取 Megatron-LM 的根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import argparse
from pathlib import Path
import math
from typing import Optional, Callable

import transformers

from pathlib import Path
import torch

from megatron.training.arguments import parse_args
from megatron.training.checkpointing import get_checkpoint_tracker_filename
from multimodal_args import add_multimodal_extra_args as extra_args_provider

class VisitTracker(dict):
    tracked_items = set()

    def __getitem__(self, item):
        self.tracked_items.add(item)
        return super().__getitem__(item)

    def untouched(self):
        untouched = []
        for k in self.keys():
            if k not in self.tracked_items:
                untouched.append(k)
        return untouched


def extra_args_provider_run_dataloader(group):
    extra_args_provider(group)
    group.add_argument('--source-ckpt', type=Path, required=True)
    group.add_argument('--load-vision', action=argparse.BooleanOptionalAction, default=True)
    group.add_argument('--load-language', action=argparse.BooleanOptionalAction, default=True)
    return group


class VisitTracker(dict):
    tracked_items = set()

    def __getitem__(self, item):
        self.tracked_items.add(item)
        return super().__getitem__(item)

    def untouched(self):
        untouched = []
        for k in self.keys():
            if k not in self.tracked_items:
                untouched.append(k)
        return untouched


def split_tensors(
    sd: dict[str, torch.Tensor],
    param_key: str,
    original_tp: int,
    target_tp: int,
    current_tp: int,
    slice_dim: Optional[int] = None,
    split_fn: Optional[Callable] = None,
):

    cnt = target_tp // original_tp
    tensor = sd[param_key]
    if slice_dim is not None:
        return torch.chunk(tensor, cnt, dim=slice_dim)[current_tp % cnt].clone()
    assert split_fn is not None
    return split_fn(tensor, cnt, current_tp % cnt)


def split_glu(sd, cnt, idx, dim=0):
    return torch.cat((
        sd.chunk(dim=dim, chunks=2)[0].chunk(cnt, dim=dim)[idx].clone(),
        sd.chunk(dim=dim, chunks=2)[1].chunk(cnt, dim=dim)[idx].clone()
    ), dim=dim)


if __name__ == "__main__":
    args = parse_args(extra_args_provider_run_dataloader, False)
    RELEASE_CKPT = args.source_ckpt
    BEGIN_OF_IMAGE_TOKEN_ID = 151339
    END_OF_IMAGE_TOKEN_ID = 151340
    print("BOI, EOI =", (BEGIN_OF_IMAGE_TOKEN_ID, END_OF_IMAGE_TOKEN_ID))
    with torch.no_grad():
        print("Loading release model")
        release_model = transformers.AutoModelForCausalLM.from_pretrained(
            RELEASE_CKPT,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        release_state_dict = release_model.state_dict()


    source_model: VisitTracker = VisitTracker(release_state_dict)
    print(f"Loaded {RELEASE_CKPT}")

    original_vision_tp = 1

    # vision configs
    vision_target_prefix = "vision_model"
    vision_projection_target_prefix = "vision_projection"
    target_vision_layers = [x for x in args.vit_layers if x]
    target_vision_tp = args.tensor_model_parallel_size

    # language configs
    original_lm_tp = 1
    original_lm_pp = 1
    target_lm_tp = args.tensor_model_parallel_size
    target_lm_layers = args.lm_layers
    original_lm_layers = 40
    assert original_lm_layers == args.num_layers, f"Expecting {original_lm_layers} layers, got {args.num_layers}"
    pre_process_pp_rank = len(target_vision_layers) - 1
    post_process_pp_rank = len(target_lm_layers) - 1
    TARGET_PP = len(target_lm_layers)
    lm_multi_query_attention = True
    lm_num_attention_heads = args.num_attention_heads
    lm_head_dim = args.kv_channels or (args.hidden_size // args.num_attention_heads)
    lm_multi_query_group_num = args.num_query_groups


    # convert configs
    LOAD_VISION = args.load_vision
    LOAD_LANGUAGE = args.load_language


    original_pp_enabled = original_lm_pp > 1
    ORIGINAL_LAYER_MAPPING: dict[int, tuple[int, int, int]] = {}  # real layer offset -> (pp_rank, vp_rank, layer_number)
    # build layer mapping
    for i in range(original_lm_layers):
        ORIGINAL_LAYER_MAPPING[i] = (0, -1, i)

    # --- vision
    if LOAD_VISION:
        print("> Loading vision parameters")

        for pp_rank, layer_count in enumerate(target_vision_layers):
            for tp in range(target_vision_tp):
                print(f"PP {pp_rank}, TP {tp}")
                
                MGT_CKPT = Path(args.save) / f'vmp_rank_{tp:02d}.pt'
                target_model = {}
            
                num_layers = 63
                assert sum(target_vision_layers) == num_layers, (
                    f"Expecting {num_layers} layers, "
                    f"got sum({target_vision_layers}) = {sum(target_vision_layers)}"
                )

                if pp_rank == 0:
                    # pre tokens
                    target_model[f'{vision_target_prefix}.class_token'] = source_model['transformer.vision.patch_embedding.cls_embedding']

                    target_model[f'{vision_target_prefix}.position_embeddings.weight'] = source_model['transformer.vision.patch_embedding.position_embedding.weight']

                    # convolution
                    target_model[f'{vision_target_prefix}.conv1.weight'] = source_model['transformer.vision.patch_embedding.proj.weight']
                
                    target_model[f'{vision_target_prefix}.conv1.bias'] = source_model['transformer.vision.patch_embedding.proj.bias']

                if pp_rank == len(target_vision_layers) - 1:
                    # downsample convolution
                    target_model[f'{vision_target_prefix}.downsample.weight'] = source_model['transformer.vision.conv.weight']
                    
                    target_model[f'{vision_target_prefix}.downsample.bias'] = source_model['transformer.vision.conv.bias']

                    # mm projector
                    target_model[f'{vision_projection_target_prefix}.encoder.linear_fc_extra.weight'] = split_tensors(
                        sd=source_model,
                        param_key='transformer.vision.linear_proj.linear_proj.weight',
                        original_tp=original_vision_tp,
                        target_tp=target_vision_tp,
                        current_tp=tp,
                        slice_dim=0,
                    )

                    target_model[f'{vision_projection_target_prefix}.encoder.linear_fc1.weight'] = torch.cat([
                        split_tensors(
                            sd=source_model,
                            param_key='transformer.vision.linear_proj.gate_proj.weight',
                            original_tp=original_vision_tp,
                            target_tp=target_vision_tp,
                            current_tp=tp,
                            slice_dim=0,
                        ),
                        split_tensors(
                            sd=source_model,
                            param_key='transformer.vision.linear_proj.dense_h_to_4h.weight',
                            original_tp=original_vision_tp,
                            target_tp=target_vision_tp,
                            current_tp=tp,
                            slice_dim=0,
                        ),
                    ], dim=0)

                    target_model[f'{vision_projection_target_prefix}.encoder.linear_fc2.weight'] = split_tensors(
                        sd=source_model,
                        param_key='transformer.vision.linear_proj.dense_4h_to_h.weight',
                        original_tp=original_vision_tp,
                        target_tp=target_vision_tp,
                        current_tp=tp,
                        slice_dim=1,
                    )
    
                    target_model[f'{vision_projection_target_prefix}.encoder.layer_norm.weight'] = source_model['transformer.vision.linear_proj.norm1.weight']
                    
                    target_model[f'{vision_projection_target_prefix}.encoder.layer_norm.bias'] = source_model['transformer.vision.linear_proj.norm1.bias']
                

                # transformer layers
                for layer_id, source_layer_id in zip(
                    range(layer_count),
                    range(sum(target_vision_layers[:pp_rank]), sum(target_vision_layers[:pp_rank + 1]))
                ):
                    print(f"copy {source_layer_id} -> {layer_id}")
                    def _copy_weight_and_bias_no_split(source_module_name: str, target_module_name):
                        for suffix in ("weight", "bias"):
                            target_model[f'{vision_target_prefix}.transformer.layers.{layer_id}.{target_module_name}.{suffix}'] = source_model[f'transformer.vision.transformer.layers.{source_layer_id}.{source_module_name}.{suffix}']
                            

                    def _copy_weight_and_bias(source_module_name: str, target_module_name, tp_dim: int, split_bias: bool):
                        for suffix in ("weight", *(["bias"] if split_bias else [])):
                            source_tensor = source_model[f'transformer.vision.transformer.layers.{source_layer_id}.{source_module_name}.{suffix}']
                            _tp_dim = 0 if suffix == "bias" else tp_dim
                            s_tp = source_tensor.shape[_tp_dim]
                            assert s_tp % target_vision_tp == 0
                            single_s_tp = s_tp // target_vision_tp
                            target_model[f'{vision_target_prefix}.transformer.layers.{layer_id}.{target_module_name}.{suffix}'] = source_tensor[:, tp * single_s_tp: (tp + 1) * single_s_tp] if _tp_dim == 1 else source_tensor[tp * single_s_tp: (tp + 1) * single_s_tp]
                            
                        if not split_bias:
                            suffix = 'bias'
                            target_model[f'{vision_target_prefix}.transformer.layers.{layer_id}.{target_module_name}.{suffix}'] = source_model[f'transformer.vision.transformer.layers.{source_layer_id}.{source_module_name}.{suffix}']
                            


                    def _copy_non_interleaved_qkv(source_module_name: str, target_module_name: str):
                        suffix = "weight"
                        source_tensor = source_model[f'transformer.vision.transformer.layers.{source_layer_id}.{source_module_name}.{suffix}']
                        q, k, v = source_tensor.chunk(3, dim=0)

                        s_tp = q.shape[0]
                        assert s_tp % target_vision_tp == 0
                        single_s_tp = s_tp // target_vision_tp
                        q, k, v = q[tp * single_s_tp: (tp + 1) * single_s_tp], k[tp * single_s_tp: (tp + 1) * single_s_tp], v[tp * single_s_tp: (tp + 1) * single_s_tp]

                        target_model[f'{vision_target_prefix}.transformer.layers.{layer_id}.{target_module_name}.{suffix}'] = torch.cat([q, k, v], dim=0)

                        suffix = 'bias'
                        source_tensor = source_model[f'transformer.vision.transformer.layers.{source_layer_id}.{source_module_name}.{suffix}']
                        q, k, v = source_tensor.chunk(3, dim=0)
                        s_tp = q.shape[0]
                        assert s_tp % target_vision_tp == 0
                        single_s_tp = s_tp // target_vision_tp
                        q, k, v = q[tp * single_s_tp: (tp + 1) * single_s_tp], k[tp * single_s_tp: (tp + 1) * single_s_tp], v[tp * single_s_tp: (tp + 1) * single_s_tp]
                        target_model[f'{vision_target_prefix}.transformer.layers.{layer_id}.{target_module_name}.{suffix}'] = torch.cat([q, k, v], dim=0)

                    # qkv projection
                    _copy_non_interleaved_qkv('attention.query_key_value', 'self_attention.linear_qkv')

                    # linear projection
                    _copy_weight_and_bias('attention.dense', 'self_attention.linear_proj', tp_dim=1, split_bias=False)

                    # layernorm1
                    _copy_weight_and_bias_no_split('input_layernorm', 'input_layernorm')

                    # layernorm2
                    _copy_weight_and_bias_no_split('post_attention_layernorm', 'pre_mlp_layernorm')

                    # mlp1
                    _copy_weight_and_bias('mlp.fc1', 'mlp.linear_fc1', tp_dim=0, split_bias=True)

                    # mlp2
                    _copy_weight_and_bias('mlp.fc2', 'mlp.linear_fc2', tp_dim=1, split_bias=False)

                torch.save(target_model, MGT_CKPT)
                print("Saved", f'vmp_rank_{tp:02d}.pt')


    def split_qkv_from_non_interleaved(sd):
        if lm_multi_query_attention:
            return sd.split(
                [
                    lm_num_attention_heads // original_lm_tp
                    * lm_head_dim,
                    lm_multi_query_group_num // original_lm_tp
                    * lm_head_dim,
                    lm_multi_query_group_num // original_lm_tp
                    * lm_head_dim,
                ],
                dim=0,
            )
        else:
            return sd.chunk(dim=0, chunks=3)

    def split_qkv(sd, cnt, idx, ):  # non interleaved kqv
        q, k, v = split_qkv_from_non_interleaved(sd)
        return torch.cat(
            (
                torch.chunk(q, cnt, dim=0)[idx],
                torch.chunk(k, cnt, dim=0)[idx],
                torch.chunk(v, cnt, dim=0)[idx],
            ),
            dim=0,
        ).clone()

    # --- language
    if LOAD_LANGUAGE:
        # convert original rope to megatron rope
        print("> Converting rope")
        with torch.no_grad():
            num_attention_heads = 32
            num_group = 2
            attention_head_size = 128
            for key, value in release_state_dict.items():
                if key.startswith("transformer.encoder") and (
                        "attention.query_key_value.weight" in key or "attention.query_key_value.bias" in key):
                    q, k, v = torch.split(value, [
                        num_attention_heads * attention_head_size, num_group * attention_head_size, num_group * attention_head_size
                    ], dim=0)
                    q = q.view(num_attention_heads, 2, attention_head_size // 4, 2, -1).transpose(2, 3).reshape(*q.shape)
                    k = k.view(num_group, 2, attention_head_size // 4, 2, -1).transpose(2, 3).reshape(*k.shape)
                    value.copy_(torch.cat([q, k, v], dim=0).detach().clone())

        print("> Loading language parameters")
        for tp in range(target_lm_tp):
            layer_offset = 0
            for pp, pp_layers in enumerate(target_lm_layers):
                print(f"TP {tp}, PP {pp}")
                pre_process = pp == pre_process_pp_rank
                post_process = pp == post_process_pp_rank
                if pp_layers == 0 and not pre_process and not post_process:
                    print("Skipping pp rank", pp)
                    continue
                
                target_model_path = Path(args.save) / f'lmp_rank_{tp:02d}'

                target_model = {}

                if pre_process:
                    # Embedding
                    source_model['transformer.embedding.word_embeddings.weight'][BEGIN_OF_IMAGE_TOKEN_ID] = source_model['transformer.vision.boi'].view(-1)
            
                    source_model['transformer.embedding.word_embeddings.weight'][END_OF_IMAGE_TOKEN_ID] = source_model['transformer.vision.eoi'].view(-1)
                    
                    target_model['embedding.word_embeddings.weight'] = split_tensors(
                        sd=source_model,
                        param_key='transformer.embedding.word_embeddings.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        slice_dim=0,
                    )
                
                    print("Finish embedding")

                # Layers
                for layer_number in range(pp_layers):
                    real_layer_offset = layer_number + layer_offset
                    original_pp_rank, original_vp_rank, original_layer_number = ORIGINAL_LAYER_MAPPING[real_layer_offset]
                    print(f"  -> real_layer_offset={real_layer_offset} original_pp_rank={original_pp_rank} original_vp_rank={original_vp_rank} original_layer_number={original_layer_number}")

                    target_model[f'decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight'] = source_model[f'transformer.encoder.layers.{original_layer_number}.input_layernorm.weight']
                    
                    target_model[f'decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight'] = source_model[f'transformer.encoder.layers.{original_layer_number}.post_attention_layernorm.weight']
                
                    target_model[f'decoder.layers.{layer_number}.self_attention.linear_qkv.weight'] = split_tensors(
                        sd=source_model,
                        param_key=f'transformer.encoder.layers.{original_layer_number}.self_attention.query_key_value.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        split_fn=split_qkv,
                    )
                    
                    target_model[f'decoder.layers.{layer_number}.self_attention.linear_qkv.bias'] = split_tensors(
                        sd=source_model,
                        param_key=f'transformer.encoder.layers.{original_layer_number}.self_attention.query_key_value.bias',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        split_fn=split_qkv,
                    )
    
                    target_model[f'decoder.layers.{layer_number}.self_attention.linear_proj.weight'] = split_tensors(
                        sd=source_model,
                        param_key=f'transformer.encoder.layers.{original_layer_number}.self_attention.dense.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        slice_dim=1,
                    )
                    
                    target_model[f'decoder.layers.{layer_number}.mlp.linear_fc1.weight'] = split_tensors(
                        sd=source_model,
                        param_key=f'transformer.encoder.layers.{original_layer_number}.mlp.dense_h_to_4h.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        split_fn=split_glu,
                    )
                    
                    target_model[f'decoder.layers.{layer_number}.mlp.linear_fc2.weight'] = split_tensors(
                        sd=source_model,
                        param_key=f'transformer.encoder.layers.{original_layer_number}.mlp.dense_4h_to_h.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        slice_dim=1,
                    )

                    print(f"Finish tp={tp} pp={pp} layer {layer_number}")

                if post_process:
                    # LM head
                    target_model['output_layer.weight']= split_tensors(
                        sd=source_model,
                        param_key='transformer.output_layer.weight',
                        original_tp=original_lm_tp,
                        target_tp=target_lm_tp,
                        current_tp=tp,
                        slice_dim=0,
                    )

                    # Final LN
                    target_model['decoder.final_layernorm.weight'] = source_model['transformer.encoder.final_layernorm.weight']
                    
                    print("Finish post process")

                torch.save(target_model, target_model_path)
                print("Saved", f'lmp_rank_{tp:02d}.pt')

                layer_offset += pp_layers

    print("Untouched SD in source model", source_model.untouched())

