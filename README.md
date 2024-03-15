# ComfyUI_VisualStylePrompting
ComfyUI Version of "Visual Style Prompting with Swapping Self-Attention"

> [!NOTE]  
> This is WIP. The WIP flag will be removed after more testing and functionality is added.

![image](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting/assets/59846140/45a2e03a-791c-4b45-bbf1-4adaae7d837b)
*origami bunny* / *a cute dog, origami style, rendered in unreal engine 5*

![image](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting/assets/59846140/1d3a6cd0-af25-42a6-a8a0-9793f85563d9)
*ControlNet Depth example.*

Implements the very basics of [Visual Style Prompting](https://github.com/naver-ai/Visual-Style-Prompting)https://github.com/naver-ai/Visual-Style-Prompting by Naver AI.

## Getting Started

Clone the reposlitory into your `custom_nodes` folder, and you'll see the node. It should be placed between your sampler and inputs like the example image.
This has currently only been tested with 1.5 based models.

- `reference_image`: The image you wish to reference,
- `conditioning_prompt`: The original positive prompt that you will transfer the style **to**
- `reference_image_prompt`: The prompt that describes the reference. You must invoke this pormpt in the conditioning prompt.
- `enabled`: Enables or disables the effect. Note that this node will still be hooked even after disabling unless you remove it.

## Notes

- Currently, this method utilized the VAE Encode & Inpaint method as it needs to iteralively denoise on each step.
Due to how this method works, you'll always get two outputs (the original next to the styled.
