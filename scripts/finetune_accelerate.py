# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import List, Literal

import torch
import tyro
from transformers import TrainingArguments
from accelerate import Accelerator

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
reserve_fraction=0.98
def reserve_gpu(gpu_id):
    torch.cuda.set_device(gpu_id)
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    reserve_bytes = int(total_mem * reserve_fraction)

    # Allocate and immediately free a tensor to reserve memory
    try:
        tensor = torch.empty((reserve_bytes // 4,), dtype=torch.float32, device=f'cuda:{gpu_id}')
    except RuntimeError as e:
        print(f"Failed to allocate on GPU {gpu_id}: {e}")
    else:
        del tensor
        torch.cuda.empty_cache()

@dataclass
class ArgsConfig:
    dataset_path: List[str]
    output_dir: str = "/tmp/gr00t"
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    batch_size: int = 32
    max_steps: int = 10000
    save_steps: int = 1000
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = True
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    resume: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False
    dataloader_num_workers: int = 8
    report_to: Literal["wandb", "tensorboard"] = "wandb"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["decord", "torchvision_av"] = "decord"
    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True
    deepspeed_config: str = ""


def main(config: ArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    if len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
        )
    else:
        single_datasets = [
            LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            ) for p in config.dataset_path
        ]
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        action_horizon=25
    )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        remove_unused_columns=False,
        deepspeed=config.deepspeed_config if config.deepspeed_config else None,
        gradient_checkpointing=True,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine", #"constant"
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        reserve_gpu(i)
        
    config = tyro.cli(ArgsConfig)

    print("\n" + "=" * 50)
    print("GR00T FINETUNING CONFIG:")
    print("=" * 50)
    for k, v in vars(config).items():
        print(f"{k}: {v}")
    print("=" * 50 + "\n")

    accelerator = Accelerator()
    accelerator.print(f"Launching training with {accelerator.num_processes} processes")
    main(config)
