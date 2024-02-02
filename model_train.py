from dataclasses import dataclass
from transformers import HfArgumentParser, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_from_disk
from peft import get_peft_model, TaskType, LoraConfig

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


    ### Data
    dataset = load_from_disk(data_args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    def tokenize_function(data):
        outputs = tokenizer(data['text'], truncation=True, max_length=model_args.max_length)
        # truncation=True : 문장 잘림 허용, max_length 보다 문장이 길 경우, max_length까지만 남김
        return outputs
    
    def collate_fn(data):
        examples_batch = tokenizer.pad(data, padding="longest", return_tensors="pt")
        examples_batch['labels'] = examples_batch['input_ids']
        return examples_batch

    train_dataset = dataset['train']
    eval_dataset = dataset['valid']

    max_train_samples = len(train_dataset)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    max_eval_samples = len(eval_dataset)
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    remove_column_keys = train_dataset.features.keys()

    train_dataset_tokenized = train_dataset.map(tokenize_function,
                                                batched=True,
                                                remove_columns=remove_column_keys)

    eval_data_tokenized = eval_dataset.map(tokenize_function,
                                           batched=True,
                                           remove_columns=remove_column_keys)
    

    ### Model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, return_dict=True)
    peft_config = LoraConfig(
        r=peft_lora_args.r,
        lora_alpha=peft_lora_args.lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=peft_lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
        )
    
    model = get_peft_model(model, peft_config)


    ### Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=eval_data_tokenized,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )

    model.config.use_cache = False
    trainer.train()

    model.eval()
    model.config.use_cache = True