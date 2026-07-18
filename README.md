# Adversarial Tokenmaxxing

Same-length, lookalike text that burns **more** BPE tokens — a small demo of how tokenizers actually chunk strings, and why that matters for cost.

Single-character substitutions (leet digits, Cyrillic homoglyphs, fullwidth forms, case flips) push text off the tokenizer’s learned merges. Character count stays fixed; token count goes up.

Open the site from this repo’s GitHub Pages deployment.
