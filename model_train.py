from dataclasses import dataclass

@dataclass
class ModelArguments:
    model_name_or_path: str = None
    max_length: int = 256