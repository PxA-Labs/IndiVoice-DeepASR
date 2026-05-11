import torch
import os
import argparse
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import sys
import os

# Add the parent directory to sys.path to allow 'from src.utils' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import prepare_dataset, DataCollatorSpeechSeq2SeqWithPadding

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/whisper-medium")
    parser.add_argument("--train_manifest", type=str, default="data/processed/svarah_manifest.json")
    parser.add_argument("--val_manifest", type=str, default="data/processed/svarah_manifest.json")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/models/whisper-indian-lora")
    parser.add_argument("--batch_size", type=int, default=4) # Reduced for VRAM safety
    parser.add_argument("--grad_accum", type=int, default=4) # Increased to keep effective batch size
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--hub_model_id", type=str, default=None)
    args = parser.parse_args()

    # 1. Device Setup for DDP
    # IMPORTANT: Do not use device_map="auto" with accelerate launch
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    
    print(f"[INFO] Rank {local_rank} starting on {device}")

    # 2. Load Processor and Model
    processor = WhisperProcessor.from_pretrained(args.model_name)
    
    # Check if a checkpoint exists for resumption
    last_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(args.output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
            print(f"[INFO] Found checkpoint: {last_checkpoint}. Resuming from here.")

    # Load base model in 8-bit to save VRAM
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name, 
        load_in_8bit=True,
        device_map={"": device} # Force model to specific GPU handled by accelerate
    )

    model = prepare_model_for_kbit_training(model)

    # 3. Handle LoRA / Resumption
    if last_checkpoint:
        print(f"[LOG] Loading existing adapters from {last_checkpoint}...")
        
        # Compatibility Fix: Strip 'alora_invocation_tokens' if it exists in config
        import json
        config_path = os.path.join(last_checkpoint, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Definitive Fix: Only keep keys that the current LoraConfig version supports
            import inspect
            from peft import LoraConfig
            valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
            
            new_config = {k: v for k, v in config_data.items() if k in valid_keys}
            
            if len(new_config) < len(config_data):
                with open(config_path, "w") as f:
                    json.dump(new_config, f, indent=4)
                print(f"[FIX] Sanitized adapter_config.json. Kept {len(new_config)} keys, removed {len(config_data) - len(new_config)} experimental ones.")

        model = PeftModel.from_pretrained(model, last_checkpoint, is_trainable=True)
    else:
        print("[LOG] Initializing new LoRA adapters...")
        config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.05, 
            bias="none"
        )
        model = get_peft_model(model, config)

    # 4. Load Datasets
    print("Loading datasets...")
    if args.train_manifest == args.val_manifest:
        print(f"[INFO] Single manifest detected. Processing and splitting...")
        full_dataset = prepare_dataset(args.train_manifest, processor.feature_extractor, processor.tokenizer)
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        train_dataset = prepare_dataset(args.train_manifest, processor.feature_extractor, processor.tokenizer)
        val_dataset = prepare_dataset(args.val_manifest, processor.feature_extractor, processor.tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Dynamic strategy handling for different transformer versions
    eval_param = "eval_strategy" if hasattr(Seq2SeqTrainingArguments, "eval_strategy") else "evaluation_strategy"
    
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_steps": 500,
        "max_steps": args.max_steps,
        "gradient_checkpointing": True,
        "fp16": True,
        eval_param: "steps",
        "per_device_eval_batch_size": args.batch_size,
        "predict_with_generate": True,
        "generation_max_length": 225,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 25,
        "report_to": "tensorboard",
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "push_to_hub": True if args.hub_model_id else False,
        "hub_model_id": args.hub_model_id,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "ddp_find_unused_parameters": False
    }

    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_lora"))
    print(f"[SUCCESS] Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    train()
