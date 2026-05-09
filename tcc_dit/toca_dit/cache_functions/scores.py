import torch.nn.functional as F


def attn_score(cache_dic, current):
    """Attention token score used by the DiT-ToCa setting."""

    attention_score = cache_dic["attn_map"][-1][current["layer"]].sum(dim=1)
    return F.normalize(attention_score, dim=1, p=2)
