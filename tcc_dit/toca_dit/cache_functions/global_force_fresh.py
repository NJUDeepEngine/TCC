def global_force_fresh(cache_dic, current):
    """Return whether the current ToCa step recomputes all tokens."""

    if cache_dic["force_fresh"] != "global":
        raise ValueError("This release exposes the global force-fresh strategy used in the paper.")

    first_step = current["step"] == current["num_steps"] - 1
    fresh_threshold = cache_dic["fresh_threshold"] if first_step else cache_dic["cal_threshold"]
    return first_step or (current["step"] % fresh_threshold == 0)
