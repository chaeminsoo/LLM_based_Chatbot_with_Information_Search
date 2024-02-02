from dataclasses import dataclass

@dataclass
class ModelArguments:
    model_name_or_path: str = None
    max_length: int = 256

@dataclass
class PeftLoraArguments:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1