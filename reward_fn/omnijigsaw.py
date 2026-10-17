import collections
import re


def check_format(predict: str) -> bool:
    """Check the strict OmniJigsaw reasoning and answer format."""
    for tag in ("<thinking>", "</thinking>", "<answer>", "</answer>"):
        if predict.count(tag) != 1:
            return False

    pattern = re.compile(r"<thinking>.*</thinking>.*<answer>.*</answer>.*", re.DOTALL)
    if re.fullmatch(pattern, predict) is None:
        return False

    return not predict.split("</answer>", maxsplit=1)[-1].strip()


def has_excessive_ngram_repetition(
    text: str,
    ngram_size: int = 20,
    max_count: int = 3,
    min_length_for_check: int = 50,
) -> bool:
    """Return whether any word N-gram occurs more than ``max_count`` times."""
    words = text.lower().split()
    if len(words) < min_length_for_check or len(words) < ngram_size:
        return False

    ngrams = collections.Counter(
        " ".join(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)
    )
    return bool(ngrams and ngrams.most_common(1)[0][1] > max_count)


def compute_score(
    solution_str: str,
    ground_truth: list[int],
    *,
    format_score: float = 0.2,
    global_pos_weight: float = 0.5,
    continuity_weight: float = 0.5,
) -> dict[str, float | int]:
    """Compute the composite OmniJigsaw temporal-reordering reward."""
    rep_penalty = (
        -0.5
        if has_excessive_ngram_repetition(
            solution_str,
            ngram_size=20,
            max_count=3,
            min_length_for_check=50,
        )
        else 0.0
    )

    if not check_format(solution_str):
        return {
            "score": rep_penalty,
            "acc_reward": 0.0,
            "format_reward": 0.0,
            "global_reward": 0.0,
            "continuity_reward": 0.0,
            "acc": 0,
            "rep_penalty": rep_penalty,
        }

    answer_text = solution_str.split("<answer>", maxsplit=1)[-1].split("</answer>", maxsplit=1)[0].strip()
    try:
        answer_indices = [int(item.strip()) for item in answer_text.split(",") if item.strip()]
    except (TypeError, ValueError):
        answer_indices = []

    n = len(ground_truth) if isinstance(ground_truth, list) else 0
    valid_permutation = (
        n > 0
        and len(answer_indices) == n
        and len(set(answer_indices)) == n
        and min(answer_indices, default=1) >= 1
        and max(answer_indices, default=n) <= n
    )

    if not valid_permutation:
        return {
            "score": rep_penalty + float(format_score),
            "acc_reward": 0.0,
            "format_reward": float(format_score),
            "global_reward": 0.0,
            "continuity_reward": 0.0,
            "acc": 0,
            "rep_penalty": rep_penalty,
        }

    global_reward = sum(
        int(predicted == expected)
        for predicted, expected in zip(answer_indices, ground_truth, strict=True)
    ) / float(n)

    if n <= 1:
        continuity_reward = 1.0
    else:
        ground_truth_pairs = {(ground_truth[i], ground_truth[i + 1]) for i in range(n - 1)}
        predicted_pairs = {(answer_indices[i], answer_indices[i + 1]) for i in range(n - 1)}
        continuity_reward = len(ground_truth_pairs & predicted_pairs) / float(n - 1)

    acc = int(answer_indices == ground_truth)
    if not acc:
        global_reward *= 0.2
        continuity_reward *= 0.2

    acc_reward = (
        float(global_pos_weight) * global_reward
        + float(continuity_weight) * continuity_reward
    )
    format_reward = float(format_score)

    return {
        "score": rep_penalty + format_reward + acc_reward,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "global_reward": global_reward,
        "continuity_reward": continuity_reward,
        "acc": acc,
        "rep_penalty": rep_penalty,
    }
