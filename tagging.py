from typing import List


def auto_tags(text: str) -> List[str]:
    t = (text or "").lower()
    tags = []

    if not t.strip():
        return ["ukendt"]

    # simple themes (DA + EN)
    if any(w in t for w in ["reklame", "tilbud", "køb", "pris", "ring", "bestil", "sponsor", "advert", "sale", "buy now"]):
        tags.append("reklame")

    if any(w in t for w in ["radio", "nyheder", "vejret", "trafik", "breaking", "news", "weather"]):
        tags.append("radio/nyheder")

    if any(w in t for w in ["politik", "regering", "minister", "folketing", "valg", "krig", "eu", "nato", "president"]):
        tags.append("politik/samfund")

    if any(w in t for w in ["børn", "børne", "dukke", "skole", "sang", "lille", "nursery", "kids"]):
        tags.append("børn")

    if any(w in t for w in ["musik", "sang", "omkvæd", "chorus", "vers", "melodi", "guitar", "piano"]):
        tags.append("musik")

    # “weird / absurd”
    if any(w in t for w in ["rum", "ufo", "alien", "magisk", "mærkelig", "absurd", "robot", "spøgelse", "space"]):
        tags.append("weird")

    return sorted(set(tags)) if tags else ["dialog"]
