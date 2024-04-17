import torch

EMPTY_PROMPT = ""


def copy_cond(cond: list[tuple[torch.Tensor, dict]], device="cpu") -> list[list[torch.Tensor | dict]]:
    cond_copy = [[]]
    t_copy = cond_copy[0]
    for t in cond:
        t_copy.append(t[0].detach().clone().to(device=device))
        if len(t) > 1 and isinstance(t[1], dict):
            t_copy.append(t[1].copy())
            if "pooled_output" in t[1]:
                t_copy[1]["pooled_output"] = t[1]["pooled_output"].detach().clone().to(device=device)
    return cond_copy


def cat_cond(clip, *conds: list[list[torch.Tensor | dict]], device="cpu") -> list[list[torch.Tensor | dict]]:
    output = copy_cond(conds[0], device=device)
    cond_tensors = list(map(lambda c: c[0][0].to(device=device), conds))

    chunk_size = 77
    chunks = list(map(lambda c: c.shape[1] // chunk_size, cond_tensors))
    chunk_min, chunk_max = min(chunks), max(chunks)

    if chunk_min != chunk_max:
        empty_tokens = clip.tokenize(EMPTY_PROMPT)
        empty_cond: torch.Tensor = clip.encode_from_tokens(empty_tokens, return_pooled=False)

        for i, cond in enumerate(cond_tensors):
            chunk_count = chunks[i]
            empty_pad = empty_cond.repeat(1, chunk_max - chunk_count, 1)
            cond_tensors[i] = torch.cat([cond.to(device=device), empty_pad.to(device=device)], dim=1)

    output[0][0] = torch.cat(cond_tensors).to(device=device)

    return output
