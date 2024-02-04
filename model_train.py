from datasets import load_from_disk
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, HfArgumentParser, TrainingArguments
import transformers

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

@dataclass
class ModelArguments:
    model_name_or_path: str = None
    max_length: int = 256

@dataclass
class PeftLoraArguments:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

@dataclass
class DataArguments:
    dataset_path: str = None
    max_train_samples: int = None # 디버깅 또는 빠른 train이 필요할 경우를 위해
    max_eval_samples: int = None


def main():
    parser = HfArgumentParser((
        ModelArguments,
        PeftLoraArguments,
        DataArguments,
        TrainingArguments
    ))

    model_args, peft_lora_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ### Load Quantized Model 

    model_id = model_args.model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                quantization_config=bnb_config,
                                                device_map="auto")
    ############################################################################



    ### Data prepare

    instruction_dataset = load_from_disk(data_args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = instruction_dataset.map(lambda samples: tokenizer(samples["text"],truncation=True, max_length=256), batched=True)

    ############################################################################



    ### PEFT

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r = peft_lora_args.r,
        lora_alpha = peft_lora_args.lora_alpha,
        lora_dropout = peft_lora_args.lora_dropout,
        target_modules = ["query_key_value"],
        bias = "none",
        task_type = "CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    ############################################################################



    # Train

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),    
    )

    model.config.use_cache = False
    trainer.train()
    ############################################################################

    model.push_to_hub('ChaeMs/KoRani-5.8b')

if __name__ == "__main__":
    main()