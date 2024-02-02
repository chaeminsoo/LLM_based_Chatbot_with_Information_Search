from dataclasses import dataclass
from transformers import HfArgumentParser, TrainingArguments

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

