"""Generic, domain-free text mutators for building `ErrorType`s.

Each factory returns a `Mutator`: `(sample, rng) -> Mutation | None` that edits one text field
and returns None when it does not apply (no number to change, a single sentence, ...). Domain
projects pass their own pools and patterns (the sentences to insert, the phrases to swap).

    from judgekit.mutators import number_swap
    wrong_amount = ErrorType("wrong_amount", frozenset({"numbers_consistent"}), number_swap())
"""

import random
import re
from collections.abc import Sequence

from judgekit.planted import Mutation, Mutator, Sample

DEFAULT_FIELD = "explanation"

# A standalone number: 500, 1,250, 3.5 (not part of a word, an id or a date like 2024-01-05).
_NUMBER = re.compile(r"(?<![\w.,/-])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w/-]|[.,]\d)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_AUXILIARY = re.compile(
    r"\b(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should|must)( not)?\b"
)
_CANNOT = re.compile(r"\bcannot\b")


def _field(sample: Sample, field: str) -> str | None:
    text = sample.fields.get(field)
    return text if text and text.strip() else None


def _format_like(value: float, int_part: str, frac: str | None) -> str:
    decimals = len(frac) - 1 if frac else 0
    return f"{value:,.{decimals}f}" if "," in int_part else f"{value:.{decimals}f}"


def number_swap(
    field: str = DEFAULT_FIELD, factors: Sequence[float] = (0.5, 0.8, 1.25, 1.5, 2.0)
) -> Mutator:
    """Change one number by a plausible factor, keeping its format (wrong amount / count)."""
    if not factors or any(f <= 0 or f == 1 for f in factors):
        raise ValueError("factors must be positive and != 1")

    def mutate(sample: Sample, rng: random.Random) -> Mutation | None:
        text = _field(sample, field)
        if text is None:
            return None
        matches = [m for m in _NUMBER.finditer(text) if float(m.group(1).replace(",", "")) > 0]
        if not matches:
            return None
        m = rng.choice(matches)
        old = m.group(0)
        value = float(m.group(1).replace(",", "") + (m.group(2) or ""))
        for factor in rng.sample(list(factors), len(factors)):
            new = _format_like(value * factor, m.group(1), m.group(2))
            if new != old:
                break
        else:
            return None
        changed = text[: m.start()] + new + text[m.end() :]
        return Mutation(changes={field: changed}, description=f"changed {old} to {new}")

    return mutate


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def drop_sentence(field: str = DEFAULT_FIELD, match: str | None = None) -> Mutator:
    """Remove one sentence (optionally only one matching the regex `match`): an omission."""
    pattern = re.compile(match, re.IGNORECASE) if match else None

    def mutate(sample: Sample, rng: random.Random) -> Mutation | None:
        text = _field(sample, field)
        if text is None:
            return None
        parts = _sentences(text)
        if len(parts) < 2:
            return None  # dropping the only sentence would leave no explanation at all
        candidates = [i for i, s in enumerate(parts) if pattern is None or pattern.search(s)]
        if not candidates:
            return None
        i = rng.choice(candidates)
        kept = " ".join(parts[:i] + parts[i + 1 :])
        return Mutation(changes={field: kept}, description=f"dropped: {parts[i]!r}")

    return mutate


def insert_sentence(sentences: Sequence[str], field: str = DEFAULT_FIELD) -> Mutator:
    """Insert one statement from `sentences` (unsupported by the evidence): a fabrication."""
    pool = [s.strip() for s in sentences if s.strip()]
    if not pool:
        raise ValueError("insert_sentence needs at least one sentence")

    def mutate(sample: Sample, rng: random.Random) -> Mutation | None:
        text = _field(sample, field)
        if text is None:
            return None
        options = [s for s in pool if s not in text]
        if not options:
            return None
        sentence = rng.choice(options)
        parts = _sentences(text)
        at = rng.randint(0, len(parts))
        out = " ".join([*parts[:at], sentence, *parts[at:]])
        return Mutation(changes={field: out}, description=f"inserted: {sentence!r}")

    return mutate


def replace_phrase(pairs: Sequence[tuple[str, str]], field: str = DEFAULT_FIELD) -> Mutator:
    """Replace one occurrence of a phrase with its counterpart (both directions are tried):
    e.g. a term with its opposite. Whole-word, case-insensitive; keeps a leading capital."""
    clean = [(a, b) for a, b in pairs if a and b and a.lower() != b.lower()]
    if not clean:
        raise ValueError("replace_phrase needs at least one pair of different phrases")
    swaps = clean + [(b, a) for a, b in clean]

    def mutate(sample: Sample, rng: random.Random) -> Mutation | None:
        text = _field(sample, field)
        if text is None:
            return None
        hits = [
            (m, new)
            for old, new in swaps
            for m in re.finditer(rf"\b{re.escape(old)}\b", text, re.IGNORECASE)
        ]
        if not hits:
            return None
        m, new = rng.choice(hits)
        found = m.group(0)
        if found[:1].isupper():
            new = new[:1].upper() + new[1:]
        out = text[: m.start()] + new + text[m.end() :]
        return Mutation(changes={field: out}, description=f"replaced {found!r} with {new!r}")

    return mutate


def negation_flip(field: str = DEFAULT_FIELD) -> Mutator:
    """Flip one negation ('is' <-> 'is not', 'cannot' <-> 'can'): a contradiction."""

    def mutate(sample: Sample, rng: random.Random) -> Mutation | None:
        text = _field(sample, field)
        if text is None:
            return None
        spots = [(m.start(), m.end(), m.group(0)) for m in _AUXILIARY.finditer(text)]
        spots += [(m.start(), m.end(), m.group(0)) for m in _CANNOT.finditer(text)]
        if not spots:
            return None
        start, end, old = rng.choice(spots)
        if old == "cannot":
            new = "can"
        elif old.endswith(" not"):
            new = old[: -len(" not")]
        else:
            new = f"{old} not"
        out = text[:start] + new + text[end:]
        return Mutation(changes={field: out}, description=f"flipped {old!r} to {new!r}")

    return mutate
