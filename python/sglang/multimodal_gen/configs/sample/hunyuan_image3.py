from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    num_frames: int = 1
    num_inference_steps: int = 8
    guidance_scale: float = 2.5
