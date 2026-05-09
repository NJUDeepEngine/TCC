import torch
from .scores import attn_score


def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    """Return the ToCa token score tensor used by the paper setting."""

    if cache_dic["cache_type"] != "attention":
        raise ValueError("This release exposes the attention-based ToCa score used in the paper.")

    score = attn_score(cache_dic, current)

    if cache_dic["force_fresh"] == "global":
        soft_step_score = cache_dic["cache_index"][-1][current["layer"]][current["module"]].float()
        soft_step_score = soft_step_score / cache_dic["fresh_threshold"]
        score = score + cache_dic["soft_fresh_weight"] * soft_step_score

    return score.to(tokens.device)
