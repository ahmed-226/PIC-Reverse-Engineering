"""
Training script for PIC Assembly-to-C Decompiler
Fine-tunes Qwen2.5-Coder model using QLoRA on PIC assembly dataset
"""

import argparse
import os
import torch
from pathlib import Path
from datetime import datetime

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import ModelConfig, TrainConfig, DataConfig
from src.data_loader import DataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train PIC Assembly-to-C Decompiler"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to master_dataset.json"
    )
    
    # Model arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="fine_tuned_model",
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        help="Base model name"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    
    # Wandb arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="PIC-Assembly-to-C-Decompiler",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/username"
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        help="Wandb API key (or set WANDB_API_KEY env var)"
    )
    
    return parser.parse_args()


def setup_wandb(train_config: TrainConfig, model_config: ModelConfig, dataset_size: int):
    """Initialize Weights & Biases logging"""
    if not train_config.use_wandb:
        return None
    
    try:
        import wandb
        
        # Set API key if provided
        if train_config.wandb_api_key:
            os.environ["WANDB_API_KEY"] = train_config.wandb_api_key
        
        run = wandb.init(
            entity=train_config.wandb_entity,
            project=train_config.wandb_project,
            config={
                "model": model_config.base_model_name,
                "learning_rate": train_config.learning_rate,
                "max_steps": train_config.max_steps,
                "dataset_size": dataset_size,
                "lora_r": model_config.lora_r,
                "lora_alpha": model_config.lora_alpha,
                "max_seq_length": model_config.max_seq_length,
                "batch_size": train_config.per_device_train_batch_size,
                "gradient_accumulation": train_config.gradient_accumulation_steps,
            },
        )
        print(f"✓ Wandb run initialized: {run.name}")
        return run
    except Exception as e:
        print(f"⚠️  Failed to initialize Wandb: {e}")
        return None


def load_model_and_tokenizer(model_config: ModelConfig):
    """Load base model and tokenizer with 4-bit quantization"""
    print(f"Loading base model: {model_config.base_model_name}")
    print(f"Max sequence length: {model_config.max_seq_length}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_config.base_model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=model_config.dtype,
        load_in_4bit=model_config.load_in_4bit,
    )
    
    print("✓ Model loaded successfully")
    return model, tokenizer


def add_lora_adapters(model, model_config: ModelConfig):
    """Add LoRA adapters to the model"""
    print(f"Adding LoRA adapters (r={model_config.lora_r}, alpha={model_config.lora_alpha})")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config.lora_r,
        target_modules=model_config.target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias=model_config.lora_bias,
        use_gradient_checkpointing=model_config.use_gradient_checkpointing,
        random_state=model_config.random_state,
    )
    
    print("✓ LoRA adapters added")
    return model


def create_trainer(model, tokenizer, train_dataset, train_config: TrainConfig, 
                  model_config: ModelConfig):
    """Create SFTTrainer instance"""
    
    training_args = TrainingArguments(
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        learning_rate=train_config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=train_config.logging_steps,
        optim=train_config.optim,
        weight_decay=train_config.weight_decay,
        lr_scheduler_type=train_config.lr_scheduler_type,
        seed=train_config.seed,
        output_dir=train_config.output_dir,
        report_to="wandb" if train_config.use_wandb else "none",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=model_config.max_seq_length,
        args=training_args,
    )
    
    return trainer


def save_model(model, tokenizer, output_dir: str, metadata: dict = None):
    """Save fine-tuned model and metadata"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to: {output_dir}")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    if metadata:
        import json
        metadata_path = output_path / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"✓ Model saved successfully to: {output_dir}")


def main():
    """Main training pipeline"""
    args = parse_args()
    
    print("="*60)
    print("PIC Assembly-to-C Decompiler - Training")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training will be very slow.")
    else:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Create configurations
    model_config = ModelConfig(
        base_model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    train_config = TrainConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_api_key=args.wandb_api_key,
    )
    
    data_config = DataConfig(dataset_path=args.dataset)
    
    # Load and prepare dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    
    data_loader = DataLoader(data_config)
    train_dataset, val_dataset = data_loader.load_json_dataset(args.dataset)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Training examples: {len(train_dataset)}")
    print(f"  • Validation examples: {len(val_dataset)}")
    print(f"  • Total examples: {len(train_dataset) + len(val_dataset)}")
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*60)
    sample_text = train_dataset[0]['text']
    print(sample_text[:500] + ("..." if len(sample_text) > 500 else ""))
    
    # Initialize Wandb
    wandb_run = setup_wandb(train_config, model_config, len(train_dataset))
    
    # Load model
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    
    model, tokenizer = load_model_and_tokenizer(model_config)
    model = add_lora_adapters(model, model_config)
    
    # Create trainer
    print("\n" + "="*60)
    print("SETTING UP TRAINER")
    print("="*60)
    
    trainer = create_trainer(model, tokenizer, train_dataset, train_config, model_config)
    
    print(f"\nTraining Configuration:")
    print(f"  • Batch size: {train_config.per_device_train_batch_size}")
    print(f"  • Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"  • Effective batch size: {train_config.per_device_train_batch_size * train_config.gradient_accumulation_steps}")
    print(f"  • Learning rate: {train_config.learning_rate}")
    print(f"  • Max steps: {train_config.max_steps}")
    print(f"  • Warmup steps: {train_config.warmup_steps}")
    
    # Train!
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Training duration: {training_duration}")
        
        # Save model
        metadata = {
            "training_date": start_time.isoformat(),
            "training_duration_seconds": training_duration.total_seconds(),
            "dataset_size": len(train_dataset),
            "validation_size": len(val_dataset),
            "model_config": {
                "base_model": model_config.base_model_name,
                "max_seq_length": model_config.max_seq_length,
                "lora_r": model_config.lora_r,
                "lora_alpha": model_config.lora_alpha,
            },
            "training_config": {
                "learning_rate": train_config.learning_rate,
                "max_steps": train_config.max_steps,
                "batch_size": train_config.per_device_train_batch_size,
                "gradient_accumulation": train_config.gradient_accumulation_steps,
            }
        }
        
        save_model(model, tokenizer, args.output_dir, metadata)
        
        print("\n✅ Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving current model state...")
        save_model(model, tokenizer, args.output_dir)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    main()
