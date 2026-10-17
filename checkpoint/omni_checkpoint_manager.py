import json
import logging
import os
import warnings
from typing import Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fsdp_utils import get_fsdp_full_state_dict
from verl.utils.fs import local_mkdir_safe
from verl.utils.logger import log_with_rank

# Configure checkpoint logging.
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class Qwen3OmniCheckpointManager(FSDPCheckpointManager):
    """
    Checkpoint manager for Qwen3-Omni models.

    Saves the trained thinker as distributed state and optionally exports a
    full model using non-thinker weights from the source checkpoint. If full
    export is unavailable or fails, the export falls back to the thinker alone.

    Args:
        model: The FSDP-wrapped thinker module
        optimizer: Training optimizer
        lr_scheduler: Learning rate scheduler
        processing_class: Tokenizer or processor
        checkpoint_config: Checkpoint configuration
        original_model_path: Path to the original Qwen3-Omni model (for loading non-thinker weights)
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        original_model_path: str = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
            **kwargs,
        )
        self.original_model_path = original_model_path

        # Cache the full-model configuration used for delivery checkpoints.
        self._original_model_config = None
        if original_model_path:
            try:
                self._original_model_config = AutoConfig.from_pretrained(
                    original_model_path,
                    trust_remote_code=True
                )
            except Exception as e:
                log_with_rank(
                    f"Warning: Could not load original model config from {original_model_path}: {e}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
    
    def _save_full_qwen3_omni_model(self, local_path: str, thinker_state_dict: dict):
        """
        Save the complete Qwen3OmniMoe model by merging trained thinker weights
        with the original model's non-thinker components.

        Args:
            local_path: Directory to save the HF model
            thinker_state_dict: State dict of the trained thinker module
        """
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeConfig,
        )

        hf_local_path = os.path.join(local_path, "huggingface")
        os.makedirs(hf_local_path, exist_ok=True)

        if self._original_model_config is None:
            raise ValueError(
                "Cannot save full Qwen3-Omni model: original_model_path was not provided "
                "or config could not be loaded. Please provide the path to the original model."
            )

        log_with_rank(
            f"Saving full Qwen3-Omni model to {hf_local_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

        # Create the full model without allocating initialized parameters.
        with init_empty_weights():
            full_model = Qwen3OmniMoeForConditionalGeneration._from_config(
                self._original_model_config,
                torch_dtype=torch.bfloat16,
            )
        full_model.to_empty(device="cpu")

        # Preserve non-thinker components for full-model checkpoint compatibility.
        original_state_dict = {}
        if self.original_model_path:
            try:
                from safetensors import safe_open
                from glob import glob

                safetensor_files = glob(os.path.join(self.original_model_path, "*.safetensors"))
                if safetensor_files:
                    for sf_path in safetensor_files:
                        with safe_open(sf_path, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                if not key.startswith("thinker."):
                                    original_state_dict[key] = f.get_tensor(key)
                else:
                    # Fall back to PyTorch checkpoint shards.
                    pt_files = glob(os.path.join(self.original_model_path, "pytorch_model*.bin"))
                    for pt_path in pt_files:
                        state = torch.load(pt_path, map_location="cpu", weights_only=True)
                        for key, value in state.items():
                            if not key.startswith("thinker."):
                                original_state_dict[key] = value

                log_with_rank(
                    f"Loaded {len(original_state_dict)} non-thinker parameters from original model",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
            except Exception as e:
                log_with_rank(
                    f"Warning: Could not load non-thinker weights from original model: {e}. "
                    f"Only thinker weights will be saved.",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )

        # Merge trained thinker weights with preserved full-model components.
        full_state_dict = {}

        full_state_dict.update(original_state_dict)

        for key, value in thinker_state_dict.items():
            # Thinker-only checkpoints omit the full-model namespace prefix.
            if not key.startswith("thinker."):
                full_key = f"thinker.{key}"
            else:
                full_key = key
            full_state_dict[full_key] = value

        log_with_rank(
            f"Merged state dict has {len(full_state_dict)} parameters "
            f"(thinker: {len(thinker_state_dict)}, non-thinker: {len(original_state_dict)})",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

        # Reuse generation settings from the source model when available.
        generation_config = None
        if self.original_model_path:
            try:
                generation_config = GenerationConfig.from_pretrained(self.original_model_path)
            except Exception:
                pass

        if generation_config is not None:
            full_model.generation_config = generation_config
            
        # Store floating-point weights in the model's delivery precision.
        for key in list(full_state_dict.keys()):
            if full_state_dict[key].dtype == torch.float32:
                full_state_dict[key] = full_state_dict[key].to(dtype=torch.bfloat16)

        full_model.save_pretrained(hf_local_path, state_dict=full_state_dict)

        log_with_rank(
            f"Successfully saved full Qwen3-Omni model to {hf_local_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

        del full_model
        del full_state_dict
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _save_thinker_only_model(self, local_path: str, thinker_state_dict: dict):
        """
        Save a thinker-only Hugging Face checkpoint when the full model is unavailable.

        Args:
            local_path: Directory to save the HF model
            thinker_state_dict: State dict of the trained thinker module
        """
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
            Qwen3OmniMoeThinkerConfig,
        )
        from verl.utils.fsdp_utils import fsdp_version

        hf_local_path = os.path.join(local_path, "huggingface")
        os.makedirs(hf_local_path, exist_ok=True)

        # Read the thinker configuration from the unwrapped module.
        if fsdp_version(self.model) == 1:
            unwrap_model = self.model._fsdp_wrapped_module
        else:
            unwrap_model = self.model

        model_config = unwrap_model.config

        log_with_rank(
            f"Saving thinker-only model to {hf_local_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

        # Create the thinker model without allocating initialized parameters.
        with init_empty_weights():
            save_model = Qwen3OmniMoeThinkerForConditionalGeneration._from_config(
                model_config,
                torch_dtype=torch.bfloat16,
            )
        save_model.to_empty(device="cpu")

        # Reuse generation settings from the source model when available.
        generation_config = None
        if self.original_model_path:
            try:
                generation_config = GenerationConfig.from_pretrained(self.original_model_path)
            except Exception:
                pass

        if generation_config is not None and save_model.can_generate():
            save_model.generation_config = generation_config
            
        # Store floating-point weights in the model's delivery precision.
        for key in list(thinker_state_dict.keys()):
            if thinker_state_dict[key].dtype == torch.float32:
                thinker_state_dict[key] = thinker_state_dict[key].to(dtype=torch.bfloat16)

        save_model.save_pretrained(hf_local_path, state_dict=thinker_state_dict)

        log_with_rank(
            f"Successfully saved thinker-only model to {hf_local_path}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )

        del save_model

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save configured distributed state and an optional Hugging Face export.

        Full-model export combines the trained thinker with source weights;
        unavailable or failed full-model exports fall back to thinker-only.
        """
        if local_path is None:
            return

        # Track the current step for checkpoint rotation and restoration.
        self.previous_global_step = global_step

        # Remove checkpoints that exceed the configured retention limit.
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # Save distributed state using the parent manager's file layout.
        from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
        from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx
        from verl.utils.device import is_cuda_available
        from dataclasses import asdict
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPConfig
        from transformers.dynamic_module_utils import custom_object_save

        if self.should_save_model:
            assert self.model is not None
        if self.should_save_optimizer:
            assert self.optimizer is not None

        # Save rank-local model, optimizer, and runtime state.
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        # Save shared configuration and preprocessing assets once.
        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)

            # Prefer the full-model configuration over the thinker-only view.
            if self._original_model_config is not None:
                self._original_model_config.save_pretrained(hf_config_tokenizer_path)
                log_with_rank(
                    f"Saved original Qwen3-Omni config to {hf_config_tokenizer_path}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
            else:
                # Fall back to the thinker configuration.
                model_config = unwrap_model.config
                model_config.save_pretrained(hf_config_tokenizer_path)

            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)

            # Record the distributed checkpoint topology.
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        torch.distributed.barrier()

        # Export a consolidated Hugging Face checkpoint when requested.
        if self.should_save_hf_model:
            # Gather the trained thinker state on rank zero.
            thinker_state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                if self.original_model_path and self._original_model_config is not None:
                    try:
                        self._save_full_qwen3_omni_model(local_path, thinker_state_dict)
                    except Exception as e:
                        log_with_rank(
                            f"Failed to save full Qwen3-Omni model: {e}. Falling back to thinker-only.",
                            rank=self.rank,
                            logger=logger,
                            log_only_rank_0=True,
                        )
                        self._save_thinker_only_model(local_path, thinker_state_dict)
                else:
                    log_with_rank(
                        "original_model_path not provided, saving thinker-only model.",
                        rank=self.rank,
                        logger=logger,
                        log_only_rank_0=True,
                    )
                    self._save_thinker_only_model(local_path, thinker_state_dict)

                del thinker_state_dict

            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)
