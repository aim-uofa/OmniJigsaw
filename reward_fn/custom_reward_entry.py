from reward_fn.omnijigsaw import compute_score as compute_omnijigsaw_score


def compute_score(solution_str, ground_truth, **kwargs):
    """VeRL entry point for the single OmniJigsaw reward implementation."""
    return compute_omnijigsaw_score(
        solution_str,
        ground_truth,
        format_score=kwargs.get("format_score", 0.2),
        global_pos_weight=kwargs.get("global_pos_weight", 0.5),
        continuity_weight=kwargs.get("continuity_weight", 0.5),
    )
