"""
HunyuanImage-3.0 Pipeline for sglang multimodal_gen.

Loads the official model with TP transformer backbone, supporting both
image generation (via generate_image()) and text generation (via generate()).
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
from safetensors import safe_open

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _iter_safetensors_weights(model_path: str):
    """Iterate over (name, tensor) pairs from safetensors files."""
    safetensors_files = sorted(
        [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    )
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    for sf_file in safetensors_files:
        filepath = os.path.join(model_path, sf_file)
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key)


class HunyuanImage3Pipeline(ComposedPipelineBase):
    pipeline_name = "HunyuanImage3Pipeline"
    _required_config_modules = []

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load model with TP transformer backbone."""
        from transformers import GenerationConfig

        from sglang.multimodal_gen.runtime.models.hunyuan_image3.configuration_hunyuan_image_3 import (
            HunyuanImage3Config,
        )
        from sglang.multimodal_gen.runtime.models.hunyuan_image3.model import (
            TPHunyuanImage3Model,
            load_tp_weights,
        )
        from sglang.multimodal_gen.runtime.models.hunyuan_image3.modeling_hunyuan_image_3 import (
            HunyuanImage3ForCausalMM,
        )

        tp_size = server_args.tp_size
        ep_size = getattr(server_args.pipeline_config, "ep_size", 1)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        # Initialize srt distributed state for FusedMoE
        self._init_srt_distributed(tp_size, ep_size)

        # Load config
        config = HunyuanImage3Config.from_pretrained(self.model_path)

        # Create TP transformer backbone
        logger.info("Creating TP transformer backbone...")
        tp_transformer = TPHunyuanImage3Model(config)

        # Load transformer weights with TP sharding
        logger.info("Loading TP transformer weights...")
        transformer_weight_count = load_tp_weights(
            tp_transformer,
            (
                (name.replace("model.", "", 1), tensor)
                for name, tensor in _iter_safetensors_weights(self.model_path)
                if name.startswith("model.") and not any(
                    skip in name
                    for skip in [
                        "vae.", "vision_model.", "vision_aligner.",
                        "patch_embed.", "time_embed.", "final_layer.",
                        "timestep_emb.", "guidance_emb.", "timestep_r_emb.",
                        "time_embed_2.", "lm_head.",
                    ]
                )
            ),
            config,
        )
        tp_transformer = tp_transformer.to(device=device, dtype=torch.bfloat16)
        tp_transformer.eval()
        logger.info("TP transformer loaded with %d weights", transformer_weight_count)

        # Create the official model with skip_load_module to avoid creating
        # the transformer (we'll assign our TP version)
        logger.info("Creating official model (auxiliary modules)...")
        official_model = HunyuanImage3ForCausalMM(
            config, skip_load_module={"transformers"}
        )

        # Assign our TP transformer as the model backbone
        official_model.model = tp_transformer
        official_model.lm_head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Load non-transformer weights (auxiliary modules + lm_head)
        logger.info("Loading auxiliary module weights...")
        aux_state_dict = {}
        for name, tensor in _iter_safetensors_weights(self.model_path):
            if name.startswith("model."):
                continue
            aux_state_dict[name] = tensor

        missing, unexpected = official_model.load_state_dict(
            aux_state_dict, strict=False
        )
        logger.info(
            "Auxiliary weights loaded. Missing: %d, Unexpected: %d",
            len(missing),
            len(unexpected),
        )
        if missing:
            real_missing = [
                k for k in missing if not k.startswith("model.")
            ]
            if real_missing:
                logger.warning("Missing auxiliary weights: %s", real_missing[:10])

        official_model = official_model.to(device=device, dtype=torch.bfloat16)
        official_model.eval()

        # Load generation config with model-specific fields
        try:
            gen_config = GenerationConfig.from_pretrained(self.model_path)
            official_model.generation_config = gen_config
            logger.info("Loaded generation_config from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load generation_config: %s", e)

        # Load tokenizer directly (config may lack model_version field)
        logger.info("Loading tokenizer...")
        from sglang.multimodal_gen.runtime.models.hunyuan_image3.tokenization_hunyuan_image_3 import (
            HunyuanImage3TokenizerFast,
        )
        official_model._tokenizer = HunyuanImage3TokenizerFast.from_pretrained(
            self.model_path
        )

        self.official_model = official_model
        self.config = config
        self.device = device

        return {}

    def _init_srt_distributed(self, tp_size: int, ep_size: int):
        """Initialize srt distributed state for FusedMoE.

        multimodal_gen uses its own distributed setup (SP-based), so we can't
        call srt.initialize_model_parallel() directly. Instead, we create
        minimal process groups that FusedMoE needs (_MOE_EP, _MOE_TP, _TP),
        and set a minimal SRT global server_args so FusedMoE.__init__ works.
        """
        import sglang.srt.distributed.parallel_state as srt_ps
        import sglang.srt.server_args as srt_sa
        from sglang.srt.distributed.parallel_state import init_model_parallel_group

        # FusedMoE and srt distributed code access various global server_args
        # fields. SRT's real ServerArgs.__post_init__ is too heavy, so we set
        # a stub that returns False/None for any missing attribute.
        if srt_sa._global_server_args is None:

            class _SRTArgsStub:
                kt_weight_path = None
                enable_symm_mem = False

                def __getattr__(self, name):
                    return None

            srt_sa._global_server_args = _SRTArgsStub()
            logger.info("Set stub SRT global server_args for FusedMoE")

        if not torch.distributed.is_initialized():
            logger.info("torch.distributed not initialized, initializing for single process")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="tcp://127.0.0.1:29599",
                world_size=1,
                rank=0,
            )

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl"

        try:
            # TP group: all ranks in one group (for TP weight sharding)
            tp_ranks = [list(range(world_size))]
            if srt_ps._TP is None:
                srt_ps._TP = init_model_parallel_group(
                    tp_ranks, local_rank, backend,
                    use_pynccl=False, use_custom_allreduce=False,
                    use_message_queue_broadcaster=False,
                    group_name="srt_tp_for_moe",
                )

            # MoE EP: each rank is its own group (no expert parallelism)
            ep_ranks = [[r] for r in range(world_size)]
            if srt_ps._MOE_EP is None:
                srt_ps._MOE_EP = init_model_parallel_group(
                    ep_ranks, local_rank, backend,
                    use_pynccl=False, use_custom_allreduce=False,
                    group_name="srt_moe_ep",
                )

            # MoE TP: same as TP (shard expert intermediate dims across GPUs)
            if srt_ps._MOE_TP is None:
                srt_ps._MOE_TP = init_model_parallel_group(
                    tp_ranks, local_rank, backend,
                    use_pynccl=False, use_custom_allreduce=False,
                    group_name="srt_moe_tp",
                )

            logger.info(
                "Initialized srt distributed for FusedMoE: world_size=%d, rank=%d",
                world_size,
                rank,
            )
        except Exception as e:
            logger.error("Failed to initialize srt distributed: %s", e, exc_info=True)

    def create_pipeline_stages(self, server_args: ServerArgs):
        """No stages - we handle everything in forward()."""
        pass

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """Generate an image or text using the HunyuanImage-3.0 model."""
        # Check if this is a text generation request
        if getattr(batch, "text_gen_mode", False):
            return self._forward_text(batch, server_args)

        return self._forward_image(batch, server_args)

    def _forward_text(self, batch: Req, server_args: ServerArgs) -> Req:
        """Text generation (prompt enhance / TI2T)."""
        prompt = batch.sampling_params.prompt
        max_new_tokens = getattr(batch, "text_gen_max_tokens", 2048)
        image_urls = getattr(batch, "text_gen_image_urls", [])

        image = None
        if image_urls:
            try:
                import base64
                from io import BytesIO

                from PIL import Image

                url = image_urls[0]
                if url.startswith("data:"):
                    _, data = url.split(",", 1)
                    image = Image.open(BytesIO(base64.b64decode(data)))
                else:
                    import requests
                    resp = requests.get(url)
                    image = Image.open(BytesIO(resp.content))
            except Exception as e:
                logger.warning("Failed to decode image: %s", e)

        generated_text = self.generate_text(
            prompt=prompt, image=image, max_new_tokens=max_new_tokens
        )
        batch.output = [generated_text]
        return batch

    def _forward_image(self, batch: Req, server_args: ServerArgs) -> Req:
        """Image generation / editing forward.

        Follows the official HunyuanImage-3.0-Instruct generate_image() flow:
        1. think_recaption: CoT reasoning → prompt rewriting → auto ratio
        2. Image diffusion with the rewritten prompt
        """
        import numpy as np
        from PIL import Image

        prompt = batch.sampling_params.prompt
        seed = batch.seeds[0] if batch.seeds else None

        # Load input images for editing (from /v1/images/edits)
        input_images = self._load_input_images(batch)

        # Use "auto" for image_size to enable automatic aspect ratio detection
        # via the think_recaption text generation stage. Only fall back to an
        # explicit size when the user requests a specific non-auto size AND
        # there are no input images (pure t2i with forced size).
        user_size = getattr(batch.sampling_params, "size", None)
        if input_images is not None:
            image_size = "auto"
        elif user_size and isinstance(user_size, str) and "x" in user_size:
            h, w = user_size.split("x")
            image_size = (int(h), int(w))
        else:
            image_size = "auto"

        use_system_prompt = "en_unified"
        bot_task = "think_recaption"

        logger.info(
            "Generating image: prompt=%s, size=%s, has_input_image=%s",
            prompt[:50] + "..." if len(prompt) > 50 else prompt,
            image_size,
            input_images is not None,
        )

        # Configure generation parameters from the model's own generation_config
        # (loaded from generation_config.json: diff_infer_steps=50, etc.)
        gen_config = self.official_model.generation_config
        gen_config.use_system_prompt = use_system_prompt
        gen_config.bot_task = bot_task
        # Synchronize CUDA random state across all TP ranks so that
        # do_sample=True produces the same token on every rank, avoiding
        # NCCL deadlocks while preserving the sampling-based prompt
        # enhancement that the official model relies on.
        self._sync_cuda_rng(seed)

        cot_text, outputs = self.official_model.generate_image(
            prompt=prompt,
            image=input_images,
            seed=seed,
            image_size=image_size,
            use_system_prompt=use_system_prompt,
            bot_task=bot_task,
            infer_align_image_size=(input_images is not None),
        )

        if cot_text:
            logger.info("CoT/recaption: %s", str(cot_text)[:200])

        raw_images = []
        if hasattr(outputs, "images"):
            raw_images = outputs.images
        elif isinstance(outputs, list):
            raw_images = outputs
        else:
            raw_images = [outputs]

        np_images = []
        for img in raw_images:
            if isinstance(img, Image.Image):
                np_images.append(np.array(img))
            elif isinstance(img, np.ndarray):
                np_images.append(img)
            elif isinstance(img, torch.Tensor):
                np_images.append(img.cpu().numpy())
            else:
                np_images.append(img)

        batch.output = np_images
        return batch

    def _load_input_images(self, batch: Req):
        """Load input image(s) from batch for image editing.

        Returns a list of PIL Images or None for text-to-image.
        """
        from PIL import Image

        image_path = getattr(batch, "image_path", None)
        if image_path is None:
            return None

        if isinstance(image_path, str):
            image_path = [image_path]

        images = []
        for p in image_path:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", p, e)

        return images if images else None

    def _sync_cuda_rng(self, seed: Optional[int] = None):
        """Synchronize CUDA random state across all TP ranks.

        Ensures do_sample=True produces identical tokens on every rank,
        preventing NCCL deadlocks from divergent autoregressive paths.
        """
        if seed is None:
            seed = 42
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        image=None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Text generation for prompt enhancement or TI2T."""
        from sglang.multimodal_gen.runtime.models.hunyuan_image3.system_prompt import (
            get_system_prompt,
        )

        if system_prompt is None:
            system_prompt = get_system_prompt("en_unified", "think")

        model_inputs = self.official_model.prepare_model_inputs(
            prompt=prompt,
            image=image,
            mode="gen_text",
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        self._sync_cuda_rng()

        input_length = model_inputs["input_ids"].shape[1]
        outputs = self.official_model.generate(
            **model_inputs, decode_text=False, **kwargs
        )
        generated_tokens = outputs[0, input_length:]
        text = self.official_model.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        return text


EntryClass = HunyuanImage3Pipeline
