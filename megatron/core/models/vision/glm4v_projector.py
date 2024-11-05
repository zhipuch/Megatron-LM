from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, gather_from_tensor_model_parallel_region
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class GLMVisionMLP(MLP):
    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            is_expert: bool = False,
            input_size: int = None,
            layer_norm_type: str = 'LayerNorm',
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            is_expert=is_expert,
            input_size=config.hidden_size,
        )

        config.normalization=layer_norm_type
        # layer norm
        self.layer_norm = submodules.norm(config, config.hidden_size)

        # extra_projection
        self.linear_fc_extra = build_module(
            submodules.linear_fc1,
            self.config.hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='fc_extra',
        )

        # extra activation
        self.extra_activation = torch.nn.GELU()

    def forward(self, hidden_states):
        # [s, b, 4 * h/p] ?

        # print('mlp - before extra', hidden_states.shape)
        # print('mlp - config', self.config.hidden_size, self.config.ffn_hidden_size)
        # for k, v in self.state_dict().items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape)

        intermediate_parallel, bias_parallel = self.linear_fc_extra(hidden_states)
        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        output = gather_from_tensor_model_parallel_region(intermediate_parallel)
        output = self.layer_norm(output)
        output = self.extra_activation(output)
        return super().forward(output)


class MultimodalProjector(MegatronModule):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from submodules is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        submodules (MLPSubmodules): Specifies MLP submodules for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        input_size: int,
    ):
        super().__init__(config=config)

        assert submodules is not None, "MLPSubmodules must be provided"

        self.encoder = GLMVisionMLP(
            config=config,  # input not gathered, output should be gathered
            submodules=submodules,
            input_size=input_size
        )

    def forward(self, hidden_states):
        # Run encoder.
        encoder_output, encoder_output_bias = self.encoder(hidden_states)

        if encoder_output_bias is not None:
            encoder_output = encoder_output + encoder_output_bias

        return encoder_output

