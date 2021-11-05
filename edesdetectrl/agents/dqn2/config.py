import dataclasses


@dataclasses.dataclass
class DQNConfig:
    prefetch_size: int = 4
