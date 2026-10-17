import logging
import time
import torch
import copy
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from tensordict import TensorDict
from typing import Generator
from verl.utils.vllm import TensorLoRARequest
from dataclasses import asdict

logger = logging.getLogger(__name__)

class OmnivLLMRollout(vLLMRollout):
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences for a batch of prompts using Qwen3-Omni specific input format.
        """
        # Convert the VeRL batch into vLLM request dictionaries.
        idx = prompts.batch["input_ids"] # (batch_size, prompt_length)
        attention_mask = prompts.batch["attention_mask"] # (batch_size, prompt_length)
        position_ids = prompts.batch["position_ids"] # (batch_size, position_axes, prompt_length)
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch

        if "text_prompt" not in non_tensor_batch:
            raise ValueError(
                "The key 'text_prompt' is missing in non_tensor_batch. "
                "OmnivLLMRollout requires raw text prompts to handle multi-modal expansion."
            )

        if batch_size != len(non_tensor_batch["text_prompt"]):
            raise RuntimeError(
                f"Batch size mismatch: expected {batch_size}, but got {len(non_tensor_batch['text_prompt'])} prompts."
            )

        vllm_inputs = []
        
        text_prompt_list = non_tensor_batch["text_prompt"]
        multi_modal_data_list = non_tensor_batch.get("multi_modal_data", [None] * batch_size)
        mm_processor_kwargs_list = non_tensor_batch.get("mm_processor_kwargs", [None] * batch_size)
        for i in range(batch_size):
            text_prompt = text_prompt_list[i]
            mm_data = multi_modal_data_list[i]
            mm_kwargs = mm_processor_kwargs_list[i]

            input_item = {
                "prompt": text_prompt
            }

            if mm_data is not None:
                if not isinstance(mm_data, dict):
                    raise TypeError(
                        f"Expected 'multi_modal_data' to be a dictionary, but got {type(mm_data)}. "
                        f"Value: {mm_data}"
                    )
                input_item["multi_modal_data"] = mm_data

            # Disable embedded video audio unless the sample enables it explicitly.
            if mm_kwargs is None:
                mm_kwargs = {"use_audio_in_video": False}
            
            if mm_kwargs is not None:
                mm_kwargs = copy.deepcopy(mm_kwargs)
            
            input_item["mm_processor_kwargs"] = mm_kwargs

            vllm_inputs.append(input_item)

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # Generate one response for greedy decoding.
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # The trainer repeats validation inputs before generation.
            }
        else:
            kwargs = {
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.9,
                "repetition_penalty": 1.05,
                "detokenize": True,
            }

        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, 
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)
            # vLLM responses use right padding.
            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        
        # Zero position IDs at padded response tokens.
        pos_mask = response_attention_mask.to(response_position_ids.device).to(response_position_ids.dtype)
        if position_ids.dim() == 3:
            # Broadcast the 2-D response mask over multimodal position axes.
            pos_mask = pos_mask.unsqueeze(1)
        
        response_position_ids = response_position_ids * pos_mask
        
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Tensor-parallel ranks receive identical valid inputs.
        batch = TensorDict(
            {
                "prompts": idx, # (batch_size, prompt_length)
                "responses": response, # (batch_size, response_length)
                "input_ids": seq,  # Concatenate prompt and response tokens.
                "attention_mask": attention_mask, # (batch_size, prompt_length + response_length)
                "position_ids": position_ids, # (batch_size, position_axes, prompt_length + response_length)
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # The actor forward pass recomputes old log probabilities.
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    
    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        
        # Map thinker-local actor weights to full-model vLLM names.
        def key_mapper(weights_gen):
            for name, tensor in weights_gen:
                if not name.startswith("thinker."):
                    if name.startswith("model.") or name.startswith("lm_head."):
                        new_name = "thinker." + name
                        yield new_name, tensor
                    else:
                        yield name, tensor
                else:
                    yield name, tensor

        mapped_weights = key_mapper(weights)
        
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(mapped_weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights ...")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(mapped_weights)
