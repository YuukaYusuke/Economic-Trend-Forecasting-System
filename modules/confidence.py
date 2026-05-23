def prediction_confidence(predicted: float, reference: float) -> tuple[str, float, str]:
    """
    Returns: (level_id: strong|medium|weak, pct_diff, label_id)
    pct_diff = abs(predicted - reference) / reference * 100
    """
    if reference == 0:
        return "weak", 0.0, "weak"

    pct = abs(predicted - reference) / abs(reference) * 100
    if pct >= 1.5:
        level = "strong"
    elif pct >= 0.4:
        level = "medium"
    else:
        level = "weak"
    return level, pct, level
