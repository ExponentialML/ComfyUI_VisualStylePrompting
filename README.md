# ComfyUI_VisualStylePrompting
ComfyUI Version of "Visual Style Prompting with Swapping Self-Attention"

![image](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting/assets/59846140/a4272a7a-5f9a-4e34-a1b7-9cb92cf7ef71)
*credits to @pamparamm*

> [!NOTE]  
> This is WIP.
> 
> Major changes were made. Please make sure to update your workflows. An updated workflow can be found in the `workflows` directory.

Implements the very basics of [Visual Style Prompting](https://github.com/naver-ai/Visual-Style-Prompting)https://github.com/naver-ai/Visual-Style-Prompting by Naver AI.

## Getting Started

Clone the reposlitory into your `custom_nodes` folder, and you'll see the node. It should be placed between your sampler and inputs like the example image.
This has currently only been tested with 1.5 based models.

- `reference_image`: The image you wish to reference,
- `positive`: The positive prompt to use as conditioning.
- `reference_cond`: The prompt that describes the reference.
- `enabled`: Enables or disables the effect. Note that this node will still be hooked even after disabling unless you remove it.
- `denoise`: Works the same way Img2Img works, but utilized with reference and / or init images (this is experimental).
- `input_blocks`: Focuses attention on the encoder layers.
- `middle_block`: Focuses attention on the middle layers.
- `output_blocks`: Focuses attention on the decoder layers.

> [!TIP]  
> In order to get the best results, you must engineer both the positive and reference image prompts correctly. Focus on the details you want to derive from the image reference, and the details you wish to see in the output.
> 
> Based on the hero image above, the positive and prompts were as follows:
>
> `kitten art, lines and colors, 3d art, unreal engine 5 render, colors, deep colors, shading, canon 60d`, `lines and colors`.
> 
> For this generation, I only wanted the lines and colors of the reference.

## Notes

- Currently, this method utilized the VAE Encode & Inpaint method as it needs to iteralively denoise on each step.
Due to how this method works, you'll always get two outputs. To remove the reference latent from the output, simple use a Batch Index Select node.

- For legacy funcionality, please pull this [PR](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting/tree/712a41579e730c5db87d68c2dcde4e92c48ab2cc).
