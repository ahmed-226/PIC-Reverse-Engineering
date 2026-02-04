"""
Configuration module for PIC Assembly-to-C Decompiler
Centralizes all hyperparameters, model settings, and default values
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture and LoRA configuration"""
    base_model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # Auto-detect (float16 for T4)
    
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules: List[str] = None
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 42
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainConfig:
    """Training hyperparameters and settings"""
    output_dir: str = "fine_tuned_model"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 300
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    seed: int = 42
    train_val_split: float = 0.9  # 90% train, 10% validation
    
    # Wandb settings
    use_wandb: bool = False
    wandb_project: str = "PIC-Assembly-to-C-Decompiler"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference settings"""
    model_path: str = "fine_tuned_model"
    max_new_tokens: int = 512
    use_cache: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    
    # Input/Output settings
    input_file: Optional[str] = None
    input_dir: Optional[str] = None
    output_dir: str = "output"
    output_format: str = "c"  # Options: c, json, markdown
    
    # Assembly parsing settings
    strip_comments: bool = True
    preserve_labels: bool = True


@dataclass
class DataConfig:
    """Dataset and prompt configuration"""
    dataset_path: str = "master_dataset.json"
    processor: str = "PIC16F877A"
    
    # Prompt template
    instruction_template: str = (
        "Decompile the following {processor} assembly to readable C code. "
        "Rename variables meaningfully and add comments."
    )
    
    # System context rules
    system_rules: List[str] = None
    
    def __post_init__(self):
        if self.system_rules is None:
            self.system_rules = [
                "Always use Hexadecimal (e.g., 0xFF) for port values, not binary.",
                "If a specific register name is unknown, infer it from the bank selection.",
                "Add comments explaining the hardware action (e.g., // Enable Timer1)."
            ]
    
    def get_instruction(self) -> str:
        """Get formatted instruction text"""
        return self.instruction_template.format(processor=self.processor)


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
DEFAULT_DATA_CONFIG = DataConfig()
