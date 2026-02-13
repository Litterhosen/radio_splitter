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

    themes = {
        "TIME": ["tid", "altid", "aldrig", "nu", "øjeblik", "sekunder", "time", "ur", "evighed"],
        "MEMORY": ["huske", "glemme", "minde", "barndom", "dengang", "remember", "tilbage"],
        "DREAM": ["drøm", "sove", "vågne", "natten", "mørke", "dream", "night"],
        "EXISTENTIAL": ["livet", "døden", "verden", "cirkel", "sjæl", "hjerte"],
        "META": ["radio", "musik", "stemme", "voice", "lyd", "sound", "lytter"],
    }

    for theme, words in themes.items():
        if any(w in t for w in words):
            tags.append(f"THEME:{theme}")

    return sorted(set(tags)) if tags else ["dialog"]
