# Adversarial Tokenmaxxing

Generates same-length, visually-similar variants of text that use **more**
BPE tokens than the original (the inverse of minification). 

## The mechanism, briefly

BPE tokenizers merge character pairs greedily based on a fixed table
learned from training data. Common substrings collapse into one token;
anything the tokenizer hasn't seen merged before falls back to smaller
pieces, in the worst case one token per character. This tool finds
single-character substitutions (`o`→`0`, `a`→`а` [Cyrillic], etc.) that push
a string off the tokenizer's learned merge paths, character by character,
greedily picking whichever available substitution yields the largest
token-count gain at each step.

**The one invariant everything else depends on**: every substitution
replaces exactly one character with exactly one character. Nothing is ever
inserted or deleted. So `len(original_text) == len(inflated_text)` always,
and character offset `i` means "the same slot" in both strings. 

## Quickstart

```bash
pip install -r requirements.txt
python3 token_inflator.py --selftest        # verify invariants hold
python3 token_inflator.py "Hello, world!" --preset aggressive --json
```

## The four strategies

All are independently toggleable and combinable (`--strategies digits,case`, etc.):

| Strategy    | What it does                                        | Subtlety                          |
| ----------- | ---------------------------------------------------- | ---------------------------------- |
| `digits`    | Classic leetspeak: `a→4 e→3 i→1 o→0 s→5`             | Most subtle, human-readable at a glance |
| `cyrillic`  | Cyrillic homoglyphs: `а е і о р с х у ѕ`             | Nearly invisible in most fonts     |
| `fullwidth` | Fullwidth forms: `ａ ｅ ｉ ｏ ...`                      | Most disruptive to tokenization, but visually obvious (wide spacing) |
| `case`      | Flips a letter's case in place (`h`→`H`)             | Zero special characters — safest choice if a site wants to avoid any Unicode confusable concerns |

There's also `max_substitution_ratio` (0.0–1.0), a cap on what fraction of
characters may be touched — this is the natural "aggressiveness slider"
for a UI, trading inflation strength for how close the output stays to the
original.

## JSON schema (the API contract)

`InflationResult.to_json()` — this is what the website's backend/logic
should produce and what the frontend should consume:

```jsonc
{
  "meta": {
    "tokenizer": "cl100k_base",
    "strategies_used": ["digits", "cyrillic"],
    "max_substitution_ratio": 1.0
  },
  "original": {
    "text": "Hi there!",
    "char_count": 9,
    "token_count": 3,
    "tokens": [
      { "index": 0, "id": 13347, "text": "Hi", "start": 0, "end": 2, "color_index": 0 },
      { "index": 1, "id": 1070, "text": " there", "start": 2, "end": 8, "color_index": 1 },
      { "index": 2, "id": 0, "text": "!", "start": 8, "end": 9, "color_index": 2 }
    ]
  },
  "inflated": {
    "text": "H1 th3r3!",
    "char_count": 9,
    "token_count": 7,
    "tokens": [ /* same shape as above, offsets into inflated.text */ ]
  },
  "stats": {
    "inflation_ratio": 2.33,
    "chars_per_token_original": 3.0,
    "chars_per_token_inflated": 1.29,
    "substituted_char_count": 3,
    "substitution_rate": 0.33
  },
  "cost": {
    "price_per_million_tokens": 3.0,
    "original_cost_usd": 0.000009,
    "inflated_cost_usd": 0.000021,
    "cost_multiplier": 2.33
  }
}
```

`cost` is `null` unless a price was supplied — see **Cost calculator** below.

**`start`/`end`** are character offsets (Python-style, end-exclusive) into
that section's own `text` field. Because of the length-invariant above,
`original.tokens[i].start` and `inflated.tokens[j].start` are directly
comparable numbers even though they come from different tokenizations —
useful if the UI wants a "hover an original token, highlight the inflated
tokens it became" feature. That's a nice-to-have, not required for v1.

## CLI reference

```
token_inflator.py [text] [options]

  text                       Text to transform. Reads stdin if omitted.
  --strategies LIST          Comma-separated: digits,cyrillic,fullwidth,case
  --preset {subtle,aggressive,max}   Overridden by --strategies if both given
  --max-substitution-ratio F  0.0-1.0, default 1.0
  --encoding NAME             Any tiktoken encoding (cl100k_base, o200k_base, ...)
  --price-per-million F       USD per 1M tokens, adds a cost block
  --json                      Emit the full JSON schema above
  --selftest                  Run invariant checks, exit 0/1
```
