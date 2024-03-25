import comfy
import torch

from .utils.attention_functions import VisualStyleProcessor
from .utils.cond_functions import cat_cond


class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "reference_image": ("IMAGE",),
                "reference_cond": ("CONDITIONING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": True}),
                "denoise": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 1e-2}),
                "input_blocks": ("BOOLEAN", {"default": False}),
                "middle_block": ("BOOLEAN", {"default": False}),
                "output_blocks": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "init_image": ("IMAGE",),
            }
        }

    CATEGORY = "VisualStylePrompting/apply"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "positive", "negative", "latents")

    FUNCTION = "apply_visual_style_prompt"

    def get_block_choices(self, input_blocks, middle_block, output_blocks):
        block_choices_map = (
            [input_blocks, "input"],
            [middle_block, "middle"],
            [output_blocks, "output"]
        )

        block_choices = []

        for block_choice in block_choices_map:
            if block_choice[0]:
                block_choices.append(block_choice[1])

        return block_choices

    def activate_block_choice(self, key, block_choices):
        return any([block in key for block in block_choices])

    def apply_visual_style_prompt(
        self,
        model: comfy.model_patcher.ModelPatcher,
        clip,
        vae,
        reference_image,
        reference_cond,
        positive,
        negative,
        enabled,
        denoise,
        input_blocks,
        middle_block,
        output_blocks,
        init_image = None
    ):
        reference_latent = vae.encode(reference_image[:,:,:,:3])

        block_choices = self.get_block_choices(input_blocks, middle_block, output_blocks)

        for n, m in model.model.diffusion_model.named_modules():
            if m.__class__.__name__  == "CrossAttention":
                is_enabled = self.activate_block_choice(n, block_choices)

                if hasattr(m.forward, 'module_self'):
                    m.forward.enabled = is_enabled and enabled
                else:
                    processor = VisualStyleProcessor(m, enabled=is_enabled)
                    setattr(m, 'forward', processor)

        positive_cat = cat_cond(clip, reference_cond, positive)
        negative_cat = cat_cond(clip, negative, negative)

        latents = torch.zeros_like(reference_latent)
        latents = torch.cat([latents] * 2)

        if denoise < 1.0:
            latents[::1] = reference_latent[:1]
        else:
            latents[::2] = reference_latent

        denoise_mask = torch.ones_like(latents)[:, :1, ...] * denoise

        denoise_mask[::2] = 0.

        return (model, positive_cat, negative_cat, {"samples": latents, "noise_mask": denoise_mask})

NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}