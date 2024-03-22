import torch


def copy_cond(cond: list[list[torch.Tensor | dict]], device="cpu"):
    cond_copy = [[]]
    t_copy = cond_copy[0]
    for t in cond:
        t_copy.append(t[0].detach().clone().to(device=device))
        if len(t) > 1 and isinstance(t[1], dict):
            t_copy.append(t[1].copy())
            if "pooled_output" in t[1]:
                t_copy[1]["pooled_output"] = t[1]["pooled_output"].detach().clone().to(device=device)
    return cond_copy
