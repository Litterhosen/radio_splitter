from rapidfuzz.fuzz import partial_ratio


JINGLE_HINTS = [
    "tilbud", "køb", "bestil", "ring", "sms",
    "sponsor", "reklame", "nu", "kun i dag",
    "introducerer", "præsenterer",
    "visit", "buy", "sale", "limited time",
]


def jingle_score(transcript: str, dur_s: float) -> float:
    """
    Heuristisk score (0-10):
    - korte klip + "call to action" ord => højere
    - længere, normal tale => lavere
    """
    t = (transcript or "").lower().strip()
    if not t:
        return 0.0

    # base: length
    score = 0.0
    if dur_s <= 5:
        score += 2.0
    elif dur_s <= 12:
        score += 1.2
    elif dur_s <= 25:
        score += 0.6

    # keyword fuzz
    for kw in JINGLE_HINTS:
        r = partial_ratio(kw, t) / 100.0
        score += 1.5 * r

    return min(10.0, score)
