from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str = "sdf_mlp"
    in_features: int = 3
    hidden_features: int = 256
    num_layers: int = 4
    out_features: int = 1
    use_fourier_pe: bool = True
    fourier_num_bands: int = 6


@dataclass
class DatasetConfig:
    name: str = "synthetic_sdf"
    num_samples: int = 4096
    batch_size: int = 512
    num_workers: int = 0
    radius: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = False
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_cfg = ModelConfig(**data.get("model", {}))
        dataset_cfg = DatasetConfig(**data.get("dataset", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        return cls(model=model_cfg, dataset=dataset_cfg, training=training_cfg)
