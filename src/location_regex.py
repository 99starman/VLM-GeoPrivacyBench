import re
from functools import lru_cache
from typing import List, Optional

import spacy

LOCATION_PREPOSITION_REGEX = re.compile(
    r"\b(?:in|at|near|inside|outside|within|around|along|on|by|beside|outside of|just outside|overlooking)\s+(?:the\s+)?(?P<place>[A-Z][A-Za-z0-9'’/&.,\s\/-]{2,150})"
)
CAPITALIZED_PHRASE_REGEX = re.compile(
    r"\b([A-Z][\w'’\-]+(?:\s+(?:of|de|la|le|del|da|do|van|von|der|den|di|du|y|the|and|&)\s*[A-Z][\w'’\-]+){0,4})"
)
LOCATION_KEYWORD_REGEX = re.compile(
    r"\b(?:city|town|district|province|state|county|park|square|plaza|temple|church|cathedral|mosque|market|bridge|station|airport|harbor|harbour|bay|beach|river|lake|falls|valley|mount|mountain|museum|university|college|campus|stadium|arena|garden|memorial|monument|castle|palace|boulevard|street|road|avenue|canal|pier|quay|harbour)\b",
    re.IGNORECASE,
)
COORDINATE_REGEX = re.compile(
    r"(?P<lat>[-+]?\d{1,2}(?:\.\d+)?)[\s°,;,/]+(?P<lng>[-+]?\d{1,3}(?:\.\d+)?)"
)
LOCATION_PREFIXES = [
    "the city of",
    "city of",
    "the town of",
    "town of",
    "the district of",
    "district of",
    "the province of",
    "province of",
    "the state of",
    "state of",
    "the region of",
    "region of",
    "the area of",
    "area of",
    "the neighborhood of",
    "neighborhood of",
    "the neighbourhood of",
    "neighbourhood of",
    "the borough of",
    "borough of",
]

LEADING_ARTICLES = [
    "the",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "my",
    "our",
    "your",
    "their",
    "guess",
    "based",
    "given",
    "overall",
    "likely",
]
STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "guess",
    "based",
    "given",
    "overall",
    "likely",
    "specific",
    "city",
    "water",
    "park",
    "roadside",
    "little",
    "one",
    "from",
    "step",
    "final",
}
GENERIC_SINGLE_WORDS = STOPWORDS | {
    "architectural",
    "visual",
    "reasoning",
    "highly",
    "street",
    "times",
    "coast",
    "plaza",
    "bridge",
    "river",
}






@lru_cache(maxsize=1)
def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    except Exception as exc:
        raise RuntimeError(
            "Failed to load spaCy model 'en_core_web_sm'. Install via `python -m spacy download en_core_web_sm`."
        ) from exc


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.;:-/\\\"'()[]{}")


def _strip_prefix(value: str) -> str:
    lower = value.lower()
    for prefix in LOCATION_PREFIXES:
        if lower.startswith(prefix + " "):
            trimmed = value[len(prefix) + 1 :].strip()
            if trimmed:
                return trimmed
    return value


def _strip_leading_articles(value: str) -> str:
    tokens = value.split()
    while len(tokens) > 1 and tokens[0].lower() in LEADING_ARTICLES:
        tokens.pop(0)
    return " ".join(tokens) if tokens else value


def _cleanup_candidate(value: str) -> str:
    cleaned = _normalize_whitespace(value)
    cleaned = _strip_prefix(cleaned)
    cleaned = _strip_leading_articles(cleaned)
    return cleaned


def _is_viable_candidate(candidate: str) -> bool:
    if not candidate:
        return False
    candidate = candidate.strip(" ,.;:-")
    if not candidate:
        return False
    lower = candidate.lower()
    if lower in STOPWORDS:
        return False
    tokens = candidate.split()
    letters = sum(ch.isalpha() for ch in candidate)
    if letters < 3:
        return False
    if len(tokens) == 1:
        if lower in GENERIC_SINGLE_WORDS:
            return False
        if len(candidate) < 4:
            return False
    return True


def _collect_capitalized_spans(sentence: str) -> List[str]:
    spans: List[str] = []
    for match in CAPITALIZED_PHRASE_REGEX.finditer(sentence):
        span = match.group(0)
        if span:
            spans.append(span)
    return spans


def _extract_noun_phrases(sentence: str) -> List[str]:
    nlp = _load_spacy_model()
    doc = nlp(sentence)
    noun_chunks: List[str] = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if not text:
            continue
        has_propn = any(token.pos_ == "PROPN" for token in chunk)
        ent_label = chunk.root.ent_type_ if chunk.root is not None else ""
        if has_propn or ent_label in {"GPE", "LOC", "FAC"}:
            noun_chunks.append(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            noun_chunks.append(ent.text.strip())
    return noun_chunks


def _extract_from_sentence(sentence: str) -> Optional[str]:
    if not sentence:
        return None

    coord_match = COORDINATE_REGEX.search(sentence)
    if coord_match:
        lat = coord_match.group("lat")
        lng = coord_match.group("lng")
        if lat and lng:
            return f"{lat},{lng}"

    prep_match = LOCATION_PREPOSITION_REGEX.search(sentence)
    if prep_match:
        candidate = _cleanup_candidate(prep_match.group("place"))
        if _is_viable_candidate(candidate):
            return candidate

    if LOCATION_KEYWORD_REGEX.search(sentence):
        for span in _collect_capitalized_spans(sentence):
            cleaned = _cleanup_candidate(span)
            if _is_viable_candidate(cleaned):
                return cleaned

    for span in _collect_capitalized_spans(sentence):
        cleaned = _cleanup_candidate(span)
        if _is_viable_candidate(cleaned):
            return cleaned

    for np_text in _extract_noun_phrases(sentence):
        cleaned = _cleanup_candidate(np_text)
        if _is_viable_candidate(cleaned):
            return cleaned

    return None


def extract_location_name_regex(text_raw: str) -> Optional[str]:
    """Regex/heuristic-based location extractor without LLM dependency."""
    if not text_raw:
        return None

    text = text_raw.strip()
    if not text:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    prioritized = sorted(
        sentences,
        key=lambda s: (
            2 if LOCATION_KEYWORD_REGEX.search(s) else 0,
            1 if any(token in s.lower() for token in [" in ", " at ", " near ", " on ", " by ", " around "]) else 0,
            -len(s),
        ),
        reverse=True,
    )

    for sentence in prioritized:
        candidate = _extract_from_sentence(sentence)
        if candidate:
            return candidate[:180]

    fallback = _extract_from_sentence(text)
    if fallback:
        return fallback[:180]

    return None
