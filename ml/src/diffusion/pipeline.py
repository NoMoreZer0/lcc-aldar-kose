from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..evaluate.evaluator import SequenceEvaluator
from ..utils import io as io_utils
from ..utils.schema import FrameControl, FrameEntry, IdentityRef, Shot, StoryboardIndex
from ..utils.vision import crop_center
from .controlnet import ControlNetManager
from .engine import SDXLEngine
from .latent_hooks import LatentHookManager
from ..consistory.consistory import ConsiStoryGenerator, ConsiStoryParams


class StoryboardGenerationPipeline:
    def __init__(
        self,
        config: Dict,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.config = config

        model_cfg = config.get("model", {})
        consistency_cfg = config.get("consistency", {})

        engine = config.get("_engine")
        self.engine = engine or SDXLEngine(model_cfg, logger=self.logger)
        self.latent_hooks = LatentHookManager(consistency_cfg, logger=self.logger)

        self.control_manager = (
            ControlNetManager(
                self.engine.device,
                dtype=self.engine.get_dtype(),
                config=model_cfg.get("controlnet", {}),
                logger=self.logger,
            )
            if model_cfg.get("use_controlnet", True)
            else None
        )
        evaluator = config.get("_evaluator")
        self.evaluator = evaluator or SequenceEvaluator(config=config, logger=self.logger)

    def _frame_seed(self, base_seed: int, offset: int) -> int:
        return base_seed + offset * 997

    def _compute_control_maps(self, previous_image, use_flags: Dict[str, bool]):
        if not previous_image or self.control_manager is None:
            return [], FrameControl()
        control_maps = self.control_manager.compute_maps(
            previous_image,
            use_depth=use_flags.get("depth", False),
            use_pose=use_flags.get("pose", False),
            use_edge=use_flags.get("edges", False),
        )
        keys = {name for name, _ in control_maps}
        flags = FrameControl(
            depth="depth" in keys,
            pose="pose" in keys,
            edges="edges" in keys,
        )
        return control_maps, flags

    def run(
        self,
        logline: str,
        shots: List[Shot],
        output_dir: Path,
        use_controlnet: bool = True,
        base_seed: int = 12345,
    ) -> StoryboardIndex:
        output_dir = io_utils.ensure_dir(output_dir)
        started = io_utils.timestamp()
        previous_image = None
        frames: List[FrameEntry] = []

        # Check if ConsiStory integration is enabled
        if self.config.get("consistency", {}).get("use_consistory", False):
            self.logger.info("Using ConsiStory for consistent multi-frame generation.")
            prompts = [shot.prompt for shot in shots]
            params = ConsiStoryParams(
                model_id=self.config.get("model", {}).get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
                width=self.config.get("model", {}).get("width", 1024),
                height=self.config.get("model", {}).get("height", 1024),
                steps=self.config.get("model", {}).get("steps", 30),
                guidance=self.config.get("model", {}).get("guidance", 7.0),
                seed=base_seed,
                consistency_strength=self.config.get("consistency", {}).get("strength", 0.6),
                self_keep_alpha=self.config.get("consistency", {}).get("self_keep_alpha", 0.7),
                dropout=self.config.get("consistency", {}).get("dropout", 0.1),
                subject_token_top_p=self.config.get("consistency", {}).get("subject_token_top_p", 0.15),
                subject_patch_top_p=self.config.get("consistency", {}).get("subject_patch_top_p", 0.3),
                feature_injection_weight=self.config.get("consistency", {}).get("feature_injection_weight", 0.5),
                max_patches_for_correspondence=self.config.get("consistency", {}).get("max_patches_for_correspondence", 64),
                attn_save_dir=self.config.get("consistency", {}).get("attn_save_dir", None),
            )
            gen = ConsiStoryGenerator(params, logger=self.logger)
            images = gen.generate_frames(prompts)

            for i, (shot, img) in enumerate(zip(shots, images)):
                frame_filename = f"frame_{shot.frame_id:02d}.png"
                io_utils.save_image(img, output_dir / frame_filename)
                frames.append(
                    FrameEntry(
                        frame_id=shot.frame_id,
                        filename=frame_filename,
                        caption=shot.caption,
                        prompt=shot.prompt,
                        seed=base_seed + i,
                        control=FrameControl(),
                        identity_ref=IdentityRef(used=True, type="consistory"),
                    )
                )
            metrics = self.evaluator.evaluate_sequence(output_dir)
            payload = StoryboardIndex(
                logline=logline,
                run_id=output_dir.name,
                frames=frames,
                metrics=metrics,
                model={"base": "SDXL", "adapters": ["consistory"]},
                timestamps={"started": started, "finished": io_utils.timestamp()},
            )
            return payload

        # Default per-frame generation
        for index, shot in enumerate(shots):
            frame_seed = self._frame_seed(base_seed, index)

            # ControlNet for structural/compositional guidance only - no img2img
            # Identity consistency relies on consistent character descriptions in prompts
            control_kwargs: Dict[str, Any] = {}
            control_flags = FrameControl()

            if use_controlnet and self.control_manager is not None and previous_image is not None:
                control_maps, control_flags = self._compute_control_maps(
                    previous_image,
                    {
                        "depth": self.config.get("consistency", {}).get("controlnet_use", {}).get("depth", False),
                        "pose": self.config.get("consistency", {}).get("controlnet_use", {}).get("pose", False),
                        "edges": self.config.get("consistency", {}).get("controlnet_use", {}).get("edges", True),
                    },
                )
                if control_maps:
                    control_images = [img for _, img in control_maps]
                    nets = self.control_manager.get_controlnets(control_flags.depth, control_flags.pose, control_flags.edges)
                    if nets:
                        # Light ControlNet conditioning for loose compositional guidance
                        # Allows variation while maintaining general scene structure
                        strength_cfg = self.config.get("consistency", {}).get("controlnet_weight", 0.3)
                        if len(control_images) == 1:
                            scale_value = min(strength_cfg, 0.3)
                        else:
                            per_map = min(strength_cfg, 0.2)
                            scale_value = [per_map] * len(control_images)
                        control_kwargs = {"control_images": control_images, "controlnet_conditioning_scale": scale_value}
                        self.engine.set_controlnet(nets)
                    else:
                        self.engine.set_controlnet(None)
                else:
                    self.engine.set_controlnet(None)
            else:
                self.engine.set_controlnet(None)

            # Use txt2img with ControlNet for all frames
            # No img2img - identity preserved through consistent prompts + light ControlNet guidance
            image = self.engine.generate(
                prompt=shot.prompt,
                seed=frame_seed,
                **control_kwargs,
            )

            frame_filename = f"frame_{shot.frame_id:02d}.png"
            io_utils.save_image(image, output_dir / frame_filename)

            if shot.frame_id == 1:
                face_crop = crop_center(image, self.config.get("consistency", {}).get("face_crop_size", 512))
                face_crop.save(output_dir / "identity_reference.png")

            frames.append(
                FrameEntry(
                    frame_id=shot.frame_id,
                    filename=frame_filename,
                    caption=shot.caption,
                    prompt=shot.prompt,
                    seed=frame_seed,
                    control=control_flags,
                    identity_ref=IdentityRef(
                        used=bool(control_kwargs),
                        type="controlnet" if control_kwargs else "none",
                    ),
                )
            )

            previous_image = image

        metrics = self.evaluator.evaluate_sequence(output_dir)

        adapters = []
        if use_controlnet and self.control_manager is not None:
            adapters.append("controlnet")

        payload = StoryboardIndex(
            logline=logline,
            run_id=output_dir.name,
            frames=frames,
            metrics=metrics,
            model={"base": "SDXL", "adapters": adapters},
            timestamps={"started": started, "finished": io_utils.timestamp()},
        )
        return payload
