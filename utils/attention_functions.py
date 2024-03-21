from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked, attention_basic
from .style_functions import adain, concat_first, swapping_attention

ATTENTION_LAYER_CANDIDATES = [
"input_blocks.1.1.transformer_blocks.0.attn1",
"input_blocks.1.1.transformer_blocks.0.attn2",
"input_blocks.2.1.transformer_blocks.0.attn1",
"input_blocks.2.1.transformer_blocks.0.attn2",
"input_blocks.4.1.transformer_blocks.0.attn1",
"input_blocks.4.1.transformer_blocks.0.attn2",
"input_blocks.5.1.transformer_blocks.0.attn1",
"input_blocks.5.1.transformer_blocks.0.attn2",
"input_blocks.7.1.transformer_blocks.0.attn1",
"input_blocks.7.1.transformer_blocks.0.attn2",
"input_blocks.8.1.transformer_blocks.0.attn1",
"input_blocks.8.1.transformer_blocks.0.attn2",
"middle_block.1.transformer_blocks.0.attn1",
"middle_block.1.transformer_blocks.0.attn2",
"output_blocks.3.1.transformer_blocks.0.attn1",
"output_blocks.3.1.transformer_blocks.0.attn2",
"output_blocks.4.1.transformer_blocks.0.attn1",
"output_blocks.4.1.transformer_blocks.0.attn2",
"output_blocks.5.1.transformer_blocks.0.attn1",
"output_blocks.5.1.transformer_blocks.0.attn2",
"output_blocks.6.1.transformer_blocks.0.attn1",
"output_blocks.6.1.transformer_blocks.0.attn2",
"output_blocks.7.1.transformer_blocks.0.attn1",
"output_blocks.7.1.transformer_blocks.0.attn2",
"output_blocks.8.1.transformer_blocks.0.attn1",
"output_blocks.8.1.transformer_blocks.0.attn2",
"output_blocks.9.1.transformer_blocks.0.attn1",
"output_blocks.9.1.transformer_blocks.0.attn2",
"output_blocks.10.1.transformer_blocks.0.attn1",
"output_blocks.10.1.transformer_blocks.0.attn2",
"output_blocks.11.1.transformer_blocks.0.attn1",
"output_blocks.11.1.transformer_blocks.0.attn2"
]

class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        keys_scale: float = 1.0,
        enabled: bool = True, 
        enabled_animatediff: bool = False,
        adain_queries: bool = True,
        adain_keys: bool = True,
        adain_values: bool = False 
    ):
        self.module_self = module_self
        self.keys_scale = keys_scale
        self.enabled = enabled
        self.enabled_animatediff = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    def visual_style_forward(self, x, context, value, mask=None):
        q = self.module_self.to_q(x)
        context = default(context, x)
        k = self.module_self.to_k(context)
        if value is not None:
            v = self.module_self.to_v(value)
            del value
        else:
            v = self.module_self.to_v(context)

        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)
            
            k = concat_first(k, -2, self.keys_scale)
            v = concat_first(v, -2)

        if mask is None:
            out = optimized_attention(q, k, v, self.module_self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.module_self.heads, mask)
        return self.module_self.to_out(out)

    # TODO
    def visual_style_forward_ad(self, x, context=None, value=None, mask=None):
        q = module_self.to_q(x)
        context = default(context, x)
        k = module_self.to_k(context)
        if value is not None:
            v = module_self.to_v(value)
            del value
        else:
            v = module_self.to_v(context)
        
        # apply custom scale by multiplying k by scale factor
        if module_self.scale is not None:
            k *= module_self.scale
        
        # apply scale mask, if present
        if scale_mask is not None:
            k *= scale_mask
        
        if self.enable_animatediff:
            k, v = swapping_attention(key, value)
            
        out = attention_basic(q, k, v, module_self.heads, mask)
        return self.to_out(out)