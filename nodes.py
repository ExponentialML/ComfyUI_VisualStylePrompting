import comfy
import torch
import nodes

from .utils.attention_functions import VisualStyleProcessor, ATTENTION_LAYER_CANDIDATES

class ApplyVisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE", ),
                "reference_image": ("IMAGE",),
                "conditioning_prompt": ("CONDITIONING",),
                "reference_image_prompt": ("CONDITIONING",),
                "negative_prompt": ("CONDITIONING", ),
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
    RETURN_TYPES = ("MODEL", "CONDITIONING","CONDITIONING", "LATENT")
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
            if block_choice[0] == True:
                block_choices.append(block_choice[1])

        return block_choices
    
    def activate_block_choice(self, key, block_choices):
        return any([block in key for block in block_choices])

    def apply_visual_style_prompt(
        self, 
        model: comfy.model_patcher.ModelPatcher, 
        vae,
        reference_image, 
        conditioning_prompt, 
        reference_image_prompt,
        negative_prompt, 
        enabled,
        denoise,
        input_blocks,
        middle_block,
        output_blocks,
        init_image = None
    ):
        self.model = model
        reference_latent = vae.encode(reference_image[:,:,:,:3])
        
        block_choices = self.get_block_choices(input_blocks, middle_block, output_blocks)

        for n, m in model.model.diffusion_model.named_modules():
            if m.__class__.__name__  == "CrossAttention": 
                processor = VisualStyleProcessor(m, enabled=self.activate_block_choice(n, block_choices))
                setattr(m, 'forward', processor.visual_style_forward)
        
        conditioning_prompt[0][0] = torch.cat([reference_image_prompt[0][0], conditioning_prompt[0][0]])
        negative_prompt[0][0] = torch.cat([negative_prompt[0][0]] * 2)

        latents = torch.zeros_like(reference_latent) 
        latents = torch.cat([latents] * 2)
        
        if denoise < 1.0:
            latents[::1] = reference_latent[:1]
        else:
            latents[::2] = reference_latent

        denoise_mask = torch.ones_like(latents)[:, :1, ...] * denoise

        denoise_mask[::2] = 0.

        return (model, conditioning_prompt, negative_prompt, {"samples": latents, "noise_mask": denoise_mask})    

NODE_CLASS_MAPPINGS = {"ApplyVisualStyle": ApplyVisualStyle}

NODE_DISPLAY_NAME_MAPPINGS = {"ApplyVisualStyle": "Apply Visual Style Prompting"}