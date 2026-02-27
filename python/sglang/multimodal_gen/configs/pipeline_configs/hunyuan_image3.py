from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class HunyuanImage3PipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I
    prefer_tp: bool = True
    dit_precision: str = "bf16"
    diff_infer_steps: int = 8
    diff_guidance_scale: float = 2.5
    flow_shift: float = 3.0
    ep_size: int = 1
