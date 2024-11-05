# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Multimodal tokenizer."""
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
# Mark tokens that will be ignored in the loss function with this value.
# Same ignore_index in https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html



@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool


class MultimodalTokenizer(MegatronTokenizer):
    """Multimodal Tokenizer."""

    def __init__(self, tokenizer: MegatronTokenizer, prompt_format: str, special_tokens: List[str]):
        """
        Tokenizer with a support for non-text inputs.

        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer (MegatronTokenizer): Underlying tokenizer.
            prompt_format (str): Prompt format for the tokenizer.
            special_tokens (List[str]): Non-text tokens.
        """
        self._vocab_size = len(tokenizer)

        #num_added_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        #assert num_added_tokens == len(
        #    special_tokens
        #), f"failed to add {len(special_tokens)} special tokens; only added {num_added_tokens}"

        self._tokenizer = tokenizer

        if prompt_format == "mistral":
            # Mistral format doesn't have prefix for the assistant message.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.unk_token_id,
                custom_chat_template=mistral_custom_template,
                has_bos=True,
                has_system_role=False,
            )
        elif prompt_format == "llama3":
            # "<|start_header_id|>assistant<|end_header|>\n\n" is the prefix for assistant messages.
            self._prompt_config = PromptConfig(
                assistant_prefix_len=4,
                pad_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
                custom_chat_template=None,
                has_bos=True,
                has_system_role=True,
            )
        elif prompt_format == "chatml":
            # "<|im_start|>assistant\n" is the prefix for assistant messages,
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=None,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "glm4v":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.pad_token_id,
                custom_chat_template=None,
                has_bos=False,
                has_system_role=False,
            )
        else:
            raise NotImplementedError("unknown multimodal tokenizer type", prompt_format)

    def tokenize(self, text: Union[str, List[Dict]]):
        """Tokenize input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(text, False, True).tolist()

        return self._tokenizer.encode(text)

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "user", "content": "something"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        input_ids = [151331, 151333]
        loss_masks = [False, False]

        for message in conversation:

            loss_mask_val = False if message['role'] in ('system', 'user', 'observation') else True
            new_input_ids_all = self._tokenizer.apply_chat_template(
                [message],
                tokenize=True,
                return_dict=True,
                padding=True
            )
            new_input_ids = new_input_ids_all['input_ids'][0][2:]
            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            
            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(151336)  # EOS
        loss_masks.append(False)
        
        input_ids = input_ids[:4] + [151339, 151329, 151340] + input_ids[4:]
        loss_masks = loss_masks[:4] + [False, False, False] + loss_masks[4:]

        targets = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                targets.append(input_id)
            else:
                targets.append(-100)
        
        return np.array(input_ids), np.array(targets)


    def convert_tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def detokenize(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    @property
    def pad(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
