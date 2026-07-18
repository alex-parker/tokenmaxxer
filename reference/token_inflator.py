#!/usr/bin/env python3
"""
token_inflator.py
==================

Generates token-maximizing variants of text: same character count (never
increases), high visual similarity to the original, but deliberately more
BPE tokens under a given tokenizer.

WHY THIS WORKS
--------------
BPE tokenizers (tiktoken/GPT family, and the same family of algorithm used
by most modern LLMs) build a fixed, priority-ordered table of character-pair
merges from training data. Common substrings collapse into a single token;
anything the tokenizer hasn't seen merged before falls back to smaller
pieces. This module finds *minimal, same-length* character substitutions
(lookalike digits, lookalike letters from other scripts, case flips) that
push a string outside the tokenizer's learned merges, character by
character, choosing at each step whichever available substitution yields
the largest token-count gain.

KEY INVARIANT
-------------
Every substitution is exactly one character replacing exactly one
character. Nothing is ever inserted or deleted. This means:

    len(original_text) == len(inflated_text)
    original_text[i] and inflated_text[i] occupy "the same slot" for all i

USAGE
-----
As a library:

    from token_inflator import inflate, Strategy

    result = inflate("Hello, world!", strategies=[Strategy.DIGITS, Strategy.CYRILLIC])
    print(result.inflated_text)
    print(result.to_json())

As a CLI:

    python3 token_inflator.py "Hello, world!" --preset aggressive
    echo "some text" | python3 token_inflator.py --json
    python3 token_inflator.py --selftest

See README.md for the full JSON schema and website build spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken

__all__ = [
    "Strategy",
    "Token",
    "InflationResult",
    "inflate",
    "tokenize_with_spans",
    "estimate_cost",
    "list_available_encodings",
]

DEFAULT_ENCODING = "cl100k_base"

# Number of distinct colors a front end is expected to cycle through when
# rendering `color_index`. Purely a convention shared with the JSON schema;
# this module never renders color itself.
PALETTE_SIZE = 8


class Strategy(str, Enum):
    """Independently toggleable substitution families.

    Each is empirically verified (see README §Design Notes) to push BPE
    tokenization apart while staying visually close to the original
    character. They can be combined freely; the greedy optimizer picks
    whichever enabled candidate helps most at each position.
    """

    DIGITS = "digits"       # classic leetspeak: a->4, e->3, i->1, o->0, s->5
    CYRILLIC = "cyrillic"   # Cyrillic homoglyphs: а е і о р с х у ѕ
    FULLWIDTH = "fullwidth"  # fullwidth forms: ａ ｅ ｉ ｏ ... (most disruptive, least subtle)
    CASE = "case"           # flips the case of a letter in place


# Per-character candidate replacements, keyed by lowercase original
# character, then by strategy. Only letters with well-established
# same-glyph-family lookalikes are included; anything not in this table is
# simply never substituted (left untouched, contributing to visual fidelity).
_SUBSTITUTIONS: dict[str, dict[Strategy, list[str]]] = {
    "a": {Strategy.DIGITS: ["4"], Strategy.CYRILLIC: ["а"], Strategy.FULLWIDTH: ["ａ"]},
    "e": {Strategy.DIGITS: ["3"], Strategy.CYRILLIC: ["е"], Strategy.FULLWIDTH: ["ｅ"]},
    "i": {Strategy.DIGITS: ["1"], Strategy.CYRILLIC: ["і"], Strategy.FULLWIDTH: ["ｉ"]},
    "o": {Strategy.DIGITS: ["0"], Strategy.CYRILLIC: ["о"], Strategy.FULLWIDTH: ["ｏ"]},
    "s": {Strategy.DIGITS: ["5"], Strategy.CYRILLIC: ["ѕ"], Strategy.FULLWIDTH: ["ｓ"]},
    "c": {Strategy.CYRILLIC: ["с"], Strategy.FULLWIDTH: ["ｃ"]},
    "p": {Strategy.CYRILLIC: ["р"], Strategy.FULLWIDTH: ["ｐ"]},
    "x": {Strategy.CYRILLIC: ["х"], Strategy.FULLWIDTH: ["ｘ"]},
    "y": {Strategy.CYRILLIC: ["у"], Strategy.FULLWIDTH: ["ｙ"]},
}


def _case_candidate(ch: str) -> Optional[str]:
    """CASE strategy candidate: flip the letter's case, if it has one."""
    if ch.isalpha():
        flipped = ch.lower() if ch.isupper() else ch.upper()
        if flipped != ch:
            return flipped
    return None


def _candidates_for_char(ch: str, strategies: set[Strategy]) -> list[str]:
    """All enabled-strategy replacement candidates for a single character."""
    out: list[str] = []
    low = ch.lower()
    table = _SUBSTITUTIONS.get(low)
    if table:
        for strat in (Strategy.DIGITS, Strategy.CYRILLIC, Strategy.FULLWIDTH):
            if strat in strategies:
                out.extend(table.get(strat, []))
    if Strategy.CASE in strategies:
        flipped = _case_candidate(ch)
        if flipped:
            out.append(flipped)
    return out


@dataclass
class Token:
    """One tokenizer token, with the character span it occupies in its
    parent string, ready for a front end to wrap in a colored <span>."""

    index: int
    id: int
    text: str
    start: int          # character offset, inclusive
    end: int             # character offset, exclusive
    color_index: int     # index % PALETTE_SIZE; front end maps this to a color

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "id": self.id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "color_index": self.color_index,
        }


def tokenize_with_spans(text: str, enc: "tiktoken.Encoding") -> list[Token]:
    """Tokenize `text` and return each token with its exact character span.

    Uses decode_single_token_bytes (raw bytes, no UTF-8 re-decoding) rather
    than decoding each token to a string in isolation. Decoding a lone
    token's ID to text can produce mangled output when a token boundary
    falls inside a multi-byte UTF-8 character (common with the homoglyphs
    this tool inserts) -- byte-level accumulation avoids that failure mode
    entirely.
    """
    token_ids = enc.encode(text)

    # Map every byte offset in the UTF-8 encoding of `text` back to a
    # character offset, so we can convert token byte-spans to char-spans.
    byte_to_char: list[int] = []
    for i, ch in enumerate(text):
        byte_to_char.extend([i] * len(ch.encode("utf-8")))
    byte_to_char.append(len(text))  # sentinel for the end-of-string offset

    tokens: list[Token] = []
    byte_pos = 0
    for idx, tid in enumerate(token_ids):
        raw = enc.decode_single_token_bytes(tid)
        start_char = byte_to_char[byte_pos]
        end_byte = byte_pos + len(raw)
        end_char = byte_to_char[end_byte] if end_byte < len(byte_to_char) else len(text)
        tokens.append(
            Token(
                index=idx,
                id=tid,
                text=text[start_char:end_char],
                start=start_char,
                end=end_char,
                color_index=idx % PALETTE_SIZE,
            )
        )
        byte_pos = end_byte
    return tokens


@dataclass
class InflationResult:
    original_text: str
    inflated_text: str
    original_tokens: list[Token]
    inflated_tokens: list[Token]
    strategies_used: list[str]
    max_substitution_ratio: float
    encoding_name: str
    substituted_positions: list[int] = field(default_factory=list)
    price_per_million_tokens: Optional[float] = None

    # --- derived stats -----------------------------------------------
    @property
    def char_count(self) -> int:
        return len(self.original_text)

    @property
    def original_token_count(self) -> int:
        return len(self.original_tokens)

    @property
    def inflated_token_count(self) -> int:
        return len(self.inflated_tokens)

    @property
    def inflation_ratio(self) -> float:
        if self.original_token_count == 0:
            return 0.0
        return self.inflated_token_count / self.original_token_count

    @property
    def substitution_rate(self) -> float:
        if self.char_count == 0:
            return 0.0
        return len(self.substituted_positions) / self.char_count

    @property
    def chars_per_token_original(self) -> float:
        return self.char_count / self.original_token_count if self.original_token_count else 0.0

    @property
    def chars_per_token_inflated(self) -> float:
        return self.char_count / self.inflated_token_count if self.inflated_token_count else 0.0

    def to_json(self) -> dict:
        cost_block = None
        if self.price_per_million_tokens is not None:
            orig_cost = estimate_cost(self.original_token_count, self.price_per_million_tokens)
            infl_cost = estimate_cost(self.inflated_token_count, self.price_per_million_tokens)
            cost_block = {
                "price_per_million_tokens": self.price_per_million_tokens,
                "original_cost_usd": orig_cost,
                "inflated_cost_usd": infl_cost,
                "cost_multiplier": (infl_cost / orig_cost) if orig_cost else 0.0,
            }
        return {
            "meta": {
                "tokenizer": self.encoding_name,
                "strategies_used": self.strategies_used,
                "max_substitution_ratio": self.max_substitution_ratio,
            },
            "original": {
                "text": self.original_text,
                "char_count": self.char_count,
                "token_count": self.original_token_count,
                "tokens": [t.to_json() for t in self.original_tokens],
            },
            "inflated": {
                "text": self.inflated_text,
                "char_count": len(self.inflated_text),
                "token_count": self.inflated_token_count,
                "tokens": [t.to_json() for t in self.inflated_tokens],
            },
            "stats": {
                "inflation_ratio": self.inflation_ratio,
                "chars_per_token_original": self.chars_per_token_original,
                "chars_per_token_inflated": self.chars_per_token_inflated,
                "substituted_char_count": len(self.substituted_positions),
                "substitution_rate": self.substitution_rate,
            },
            "cost": cost_block,
        }


def estimate_cost(token_count: int, price_per_million_tokens: float) -> float:
    """Pure arithmetic: cost in USD for `token_count` tokens at a given
    $-per-million-tokens rate. Callers/front ends supply the rate; this
    module deliberately hardcodes no vendor pricing, since that changes
    over time. See README §Cost Calculator for where to source current
    rates.
    """
    return (token_count / 1_000_000) * price_per_million_tokens


def list_available_encodings() -> list[str]:
    """Encodings tiktoken can target. cl100k_base = GPT-3.5/4 family,
    o200k_base = GPT-4o/GPT-5 family. Anthropic's own tokenizer isn't
    public, so these are the closest available reference points for
    demonstrating BPE-family behavior generally -- see README for the
    caveat to surface on the site."""
    return tiktoken.list_encoding_names()


def inflate(
    text: str,
    strategies: Optional[list[Strategy]] = None,
    max_substitution_ratio: float = 1.0,
    encoding_name: str = DEFAULT_ENCODING,
    price_per_million_tokens: Optional[float] = None,
) -> InflationResult:
    """Produce a token-maximized variant of `text`.

    Parameters
    ----------
    strategies:
        Which substitution families are allowed. Defaults to all of them.
        Pass a subset to match a UI toggle state, e.g. [Strategy.DIGITS].
    max_substitution_ratio:
        Upper bound, as a fraction of total characters (0.0-1.0), on how
        many characters may be substituted. 1.0 (default) = no cap, the
        greedy optimizer keeps going as long as it finds improving moves.
        Lower values trade inflation strength for visual subtlety -- a
        natural "aggressiveness" slider for a UI.
    encoding_name:
        Any name from list_available_encodings().
    price_per_million_tokens:
        If provided, included cost estimates in the result / JSON output.

    Returns
    -------
    InflationResult
        original_text is guaranteed unchanged; inflated_text is guaranteed
        the same length as original_text (see module docstring invariant).
    """
    if strategies is None:
        strategies = list(Strategy)
    strategy_set = set(strategies)
    enc = tiktoken.get_encoding(encoding_name)

    chars = list(text)
    changed_positions: list[int] = []
    max_changes = int(len(chars) * max_substitution_ratio)

    while len(changed_positions) < max_changes:
        base_count = len(enc.encode("".join(chars)))
        best: Optional[tuple[int, int, str]] = None  # (new_count, position, replacement)
        for i, ch in enumerate(chars):
            if i in changed_positions:
                continue
            for repl in _candidates_for_char(ch, strategy_set):
                trial = chars.copy()
                trial[i] = repl
                new_count = len(enc.encode("".join(trial)))
                if new_count > base_count and (best is None or new_count > best[0]):
                    best = (new_count, i, repl)
        if best is None:
            break
        _, pos, repl = best
        chars[pos] = repl
        changed_positions.append(pos)

    inflated_text = "".join(chars)
    original_tokens = tokenize_with_spans(text, enc)
    inflated_tokens = tokenize_with_spans(inflated_text, enc)

    return InflationResult(
        original_text=text,
        inflated_text=inflated_text,
        original_tokens=original_tokens,
        inflated_tokens=inflated_tokens,
        strategies_used=[s.value for s in strategies],
        max_substitution_ratio=max_substitution_ratio,
        encoding_name=encoding_name,
        substituted_positions=sorted(changed_positions),
        price_per_million_tokens=price_per_million_tokens,
    )


# ----------------------------------------------------------------------
# Self-test: validates the invariants a front end will rely on. A coding
# agent porting this to another language should reproduce these checks.
# ----------------------------------------------------------------------

def _selftest() -> None:
    cases = [
        "Would it be possible to assemble a scheme of typos?",
        "",
        "a",
        "12345",
        "日本語のテスト",  # non-Latin input: nothing in the substitution table should fire
        "MiXeD CaSe Text!!",
    ]
    presets = {
        "digits_only": [Strategy.DIGITS],
        "cyrillic_only": [Strategy.CYRILLIC],
        "fullwidth_only": [Strategy.FULLWIDTH],
        "case_only": [Strategy.CASE],
        "all": list(Strategy),
    }
    failures = 0
    for text in cases:
        for name, strategies in presets.items():
            result = inflate(text, strategies=strategies, price_per_million_tokens=3.0)
            ok = True
            if len(result.inflated_text) != len(result.original_text):
                print(f"FAIL[{name}] length changed for {text!r}")
                ok = False
            if result.inflated_token_count < result.original_token_count:
                print(f"FAIL[{name}] tokens *decreased* for {text!r}")
                ok = False
            # spans must cover the string exactly with no gaps/overlaps
            for label, toks, src in (
                ("original", result.original_tokens, result.original_text),
                ("inflated", result.inflated_tokens, result.inflated_text),
            ):
                cursor = 0
                for t in toks:
                    if t.start != cursor or src[t.start:t.end] != t.text:
                        print(f"FAIL[{name}] span mismatch in {label} for {text!r}: {t}")
                        ok = False
                        break
                    cursor = t.end
                if cursor != len(src):
                    print(f"FAIL[{name}] spans don't cover full {label} string for {text!r}")
                    ok = False
            if not ok:
                failures += 1
    total = len(cases) * len(presets)
    print(f"selftest: {total - failures}/{total} passed")
    sys.exit(1 if failures else 0)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

_PRESETS: dict[str, list[Strategy]] = {
    "subtle": [Strategy.DIGITS, Strategy.CYRILLIC],
    "aggressive": [Strategy.DIGITS, Strategy.CYRILLIC, Strategy.FULLWIDTH],
    "max": list(Strategy),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a token-maximized, same-length, visually-similar variant of text.",
    )
    parser.add_argument("text", nargs="?", help="Text to transform. Reads stdin if omitted.")
    parser.add_argument(
        "--strategies",
        help="Comma-separated: digits,cyrillic,fullwidth,case. Overrides --preset.",
    )
    parser.add_argument("--preset", choices=_PRESETS.keys(), default="max")
    parser.add_argument("--max-substitution-ratio", type=float, default=1.0)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING, choices=list_available_encodings())
    parser.add_argument("--price-per-million", type=float, default=None, help="USD per 1M tokens, for cost estimate.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON result instead of plain text.")
    parser.add_argument("--selftest", action="store_true", help="Run invariant checks and exit.")
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    text = args.text
    if text is None:
        text = sys.stdin.read()

    if args.strategies:
        strategies = [Strategy(s.strip()) for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = _PRESETS[args.preset]

    result = inflate(
        text,
        strategies=strategies,
        max_substitution_ratio=args.max_substitution_ratio,
        encoding_name=args.encoding,
        price_per_million_tokens=args.price_per_million,
    )

    if args.json:
        print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))
    else:
        print(result.inflated_text)
        print(
            f"\n[{result.original_token_count} -> {result.inflated_token_count} tokens, "
            f"{result.char_count} chars unchanged, "
            f"{result.inflation_ratio:.2f}x, "
            f"{len(result.substituted_positions)} chars substituted "
            f"({result.substitution_rate:.0%})]",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
