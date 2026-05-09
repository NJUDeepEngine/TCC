def fresh_ratio_scheduler(cache_dic, current):
    """Return the fresh-token ratio for the DiT-ToCa setting used in the paper."""

    if cache_dic["fresh_ratio_schedule"] != "ToCa-ddim50":
        raise ValueError("This release exposes the ToCa-ddim50 scheduler used in the paper.")

    fresh_ratio = cache_dic["fresh_ratio"]
    step = current["step"]
    num_steps = current["num_steps"]

    step_weight = 2.0
    step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

    layer_weight = -0.2
    layer_factor = 1 + layer_weight - 2 * layer_weight * current["layer"] / 27

    module_weight = 2.5
    module_time_weight = 0.6
    if current["module"] == "attn":
        module_factor = 1 - (1 - module_time_weight) * module_weight
    else:
        module_factor = 1 + module_time_weight * module_weight

    return fresh_ratio * layer_factor * step_factor * module_factor
