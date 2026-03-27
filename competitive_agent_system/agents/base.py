from __future__ import annotations

from typing import Any

import numpy as np
from transformers import PreTrainedTokenizer

from agent_system.multi_turn_rollout.utils import preprocess_batch
from verl import DataProto
from verl.protocol import extract_dataproto_via_active_mask, pad_dataproto_to_divisor, restore_dataproto_via_active_mask, unpad_dataproto


class CompetitiveBaseAgent:
    """Base agent for competitive environments where prompts are built by the game."""

    def __init__(self, name: str, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        self.name = name
        self.wg_id = wg_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    def reset(self):
        pass

    def _generate_with_llm(self, batch: DataProto, actor_rollout_wg, agent_active_mask: np.ndarray, meta_info) -> tuple[DataProto, list[str]]:
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")

        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = meta_info

        batch_input_extracted = extract_dataproto_via_active_mask(batch_input, agent_active_mask)
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input_extracted, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output_extracted = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        batch_output = restore_dataproto_via_active_mask(batch_output_extracted, agent_active_mask)

        batch = batch.union(batch_output)
        text_responses = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

        batch.non_tensor_batch["wg_id"] = np.array([self.wg_id] * len(batch), dtype=object)
        batch.non_tensor_batch["agent_active_mask"] = agent_active_mask

        return batch, text_responses

    def call(self, gen_batch: DataProto, obs_texts: list[str], actor_rollout_wg, agent_active_mask: np.ndarray, step: int) -> tuple[DataProto, list[str]]:
        obs = {
            "text": obs_texts,
            "image": None,
            "anchor": None,
        }
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        batch, text_responses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        batch.non_tensor_batch["env_step"] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
