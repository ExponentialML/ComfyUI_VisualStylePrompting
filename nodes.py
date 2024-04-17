import comfy
import torch

from .utils.attention_functions import VisualStyleProcessor
from .utils.cond_functions import copy_cond


class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE", ),
                "reference_image": ("IMAGE",),
                "reference_cond": ("CONDITIONING",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": True}),
                "denoise": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": 1e-2}),
                "input_blocks": ("BOOLEAN", {"default": False}),
                "skip_input_layers": ("INT", {"default": 24, "min": 0, "max": 48, "step": 1}),
                "middle_block": ("BOOLEAN", {"default": False}),
                "skip_middle_layers": ("INT", {"default": 1, "min": 0, "max": 2, "step": 1}),
                "output_blocks": ("BOOLEAN", {"default": True}),
                "skip_output_layers": ("INT", {"default": 24, "min": 0, "max": 72, "step": 1}),
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
        init_image = None,
        skip_input_layers=0,
        skip_middle_layers=0,
        skip_output_layers=0

    ):
        positive = copy_cond(positive)
        reference_cond = copy_cond(reference_cond)
        negative = copy_cond(negative)

        reference_latent = vae.encode(reference_image[:,:,:,:3])

        block_choices = self.get_block_choices(input_blocks, middle_block, output_blocks)


        layer_indexes = {
            "input": 0,
            "middle": 0,
            "output": 0
        }

        n_skip_per_block = {
            "input": skip_input_layers,
            "middle": skip_middle_layers,
            "output": skip_output_layers
        }

        class_names = set()
        for n, m in model.model.diffusion_model.named_modules():
            if m.__class__.__name__ == "CrossAttention":
                is_enabled = self.activate_block_choice(n, block_choices)
                if is_enabled:
                    block_name = n.split("_")[0]
                    if layer_indexes[block_name] < n_skip_per_block[block_name]:
                        is_enabled = False
                    layer_indexes[block_name] += 1
                if hasattr(m.forward, 'module_self'):
                    m.forward.enabled = is_enabled and enabled
                else:
                    processor = VisualStyleProcessor(m, enabled=is_enabled)
                    setattr(m, 'forward', processor)

        positive[0][0] = torch.cat([reference_cond[0][0], positive[0][0]])
        negative[0][0] = torch.cat([negative[0][0]] * 2)

        latents = torch.zeros_like(reference_latent)
        latents = torch.cat([latents] * 2)
        latents[0] = reference_latent

        denoise_mask = torch.ones_like(latents)[:, :1, ...] * denoise

        denoise_mask[0] = 0.

        return (model, positive, negative, {"samples": latents, "noise_mask": denoise_mask})

NODE_CLASS_MAPPINGS = {
    "ApplyVisualStyle": ApplyVisualStyle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyVisualStyle": "Apply Visual Style Prompting",
}