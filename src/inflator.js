import {
  DEFAULT_ENCODING,
  PALETTE_SIZE,
  getEncoder,
  tokenizeWithSpans,
} from "./tokenizer.js";

export { DEFAULT_ENCODING, PALETTE_SIZE };

/** Cap for interactive use — keeps inflate responsive in-browser. */
export const MAX_INPUT_CHARS = 400;

/** Radius (in characters) for local token-delta estimates. */
const WINDOW_RADIUS = 32;

export const Strategy = {
  DIGITS: "digits",
  CYRILLIC: "cyrillic",
  FULLWIDTH: "fullwidth",
  CASE: "case",
};

export const ALL_STRATEGIES = [
  Strategy.DIGITS,
  Strategy.CYRILLIC,
  Strategy.FULLWIDTH,
  Strategy.CASE,
];

export const PRESETS = {
  subtle: [Strategy.DIGITS, Strategy.CYRILLIC],
  aggressive: [Strategy.DIGITS, Strategy.CYRILLIC, Strategy.FULLWIDTH],
  max: [...ALL_STRATEGIES],
};

const SUBSTITUTIONS = {
  a: {
    [Strategy.DIGITS]: ["4"],
    [Strategy.CYRILLIC]: ["а"],
    [Strategy.FULLWIDTH]: ["ａ"],
  },
  e: {
    [Strategy.DIGITS]: ["3"],
    [Strategy.CYRILLIC]: ["е"],
    [Strategy.FULLWIDTH]: ["ｅ"],
  },
  i: {
    [Strategy.DIGITS]: ["1"],
    [Strategy.CYRILLIC]: ["і"],
    [Strategy.FULLWIDTH]: ["ｉ"],
  },
  o: {
    [Strategy.DIGITS]: ["0"],
    [Strategy.CYRILLIC]: ["о"],
    [Strategy.FULLWIDTH]: ["ｏ"],
  },
  s: {
    [Strategy.DIGITS]: ["5"],
    [Strategy.CYRILLIC]: ["ѕ"],
    [Strategy.FULLWIDTH]: ["ｓ"],
  },
  c: {
    [Strategy.CYRILLIC]: ["с"],
    [Strategy.FULLWIDTH]: ["ｃ"],
  },
  p: {
    [Strategy.CYRILLIC]: ["р"],
    [Strategy.FULLWIDTH]: ["ｐ"],
  },
  x: {
    [Strategy.CYRILLIC]: ["х"],
    [Strategy.FULLWIDTH]: ["ｘ"],
  },
  y: {
    [Strategy.CYRILLIC]: ["у"],
    [Strategy.FULLWIDTH]: ["ｙ"],
  },
};

function caseCandidate(ch) {
  if (!/\p{L}/u.test(ch)) return null;
  const flipped = ch === ch.toUpperCase() ? ch.toLowerCase() : ch.toUpperCase();
  return flipped !== ch ? flipped : null;
}

export function candidatesForChar(ch, strategies) {
  const out = [];
  const low = ch.toLowerCase();
  const table = SUBSTITUTIONS[low];
  if (table) {
    for (const strat of [Strategy.DIGITS, Strategy.CYRILLIC, Strategy.FULLWIDTH]) {
      if (strategies.has(strat)) {
        out.push(...(table[strat] ?? []));
      }
    }
  }
  if (strategies.has(Strategy.CASE)) {
    const flipped = caseCandidate(ch);
    if (flipped) out.push(flipped);
  }
  return out;
}

function estimateCost(tokenCount, pricePerMillionTokens) {
  return (tokenCount / 1_000_000) * pricePerMillionTokens;
}

function tokenCount(enc, chars) {
  return enc.encode(chars.join("")).length;
}

function windowTokenCount(enc, chars, center, radius, overridePos, overrideCh) {
  const start = Math.max(0, center - radius);
  const end = Math.min(chars.length, center + radius + 1);
  let s = "";
  for (let i = start; i < end; i++) {
    s += i === overridePos ? overrideCh : chars[i];
  }
  return enc.encode(s).length;
}

function localDelta(enc, chars, pos, repl) {
  const base = windowTokenCount(enc, chars, pos, WINDOW_RADIUS);
  const trial = windowTokenCount(enc, chars, pos, WINDOW_RADIUS, pos, repl);
  return trial - base;
}

/**
 * Exact greedy search (full-string encode every trial). Used for short inputs
 * so golden fixtures and small demos stay bit-identical to the Python reference.
 */
function inflateExact(enc, chars, strategySet, maxChanges) {
  const changedSet = new Set();
  const changedPositions = [];

  while (changedPositions.length < maxChanges) {
    const open = [];
    for (let i = 0; i < chars.length; i++) {
      if (changedSet.has(i)) continue;
      const cands = candidatesForChar(chars[i], strategySet);
      if (cands.length) open.push({ i, cands });
    }
    if (open.length === 0) break;

    const baseCount = tokenCount(enc, chars);
    let best = null;
    for (const { i, cands } of open) {
      const orig = chars[i];
      for (const repl of cands) {
        chars[i] = repl;
        const newCount = tokenCount(enc, chars);
        chars[i] = orig;
        if (newCount > baseCount && (best === null || newCount > best.newCount)) {
          best = { newCount, position: i, replacement: repl };
        }
      }
    }
    if (best === null) break;
    chars[best.position] = best.replacement;
    changedSet.add(best.position);
    changedPositions.push(best.position);
  }

  return changedPositions;
}

/**
 * Fast greedy for longer inputs:
 * 1. Score every (position, replacement) by local window token delta once
 * 2. Pop best scores; verify with one full-string encode
 * 3. On accept, bump a per-position generation (invalidates stale heap entries)
 *    and rescore only positions within WINDOW_RADIUS of the change
 */
function inflateFast(enc, chars, strategySet, maxChanges) {
  const changedSet = new Set();
  const changedPositions = [];
  const generation = new Int32Array(chars.length);
  /** @type {{ delta: number, i: number, repl: string, g: number }[]} */
  const heap = [];

  const pushMove = (i, repl) => {
    const delta = localDelta(enc, chars, i, repl);
    if (delta > 0) heap.push({ delta, i, repl, g: generation[i] });
  };

  const rescorePosition = (i) => {
    if (changedSet.has(i)) return;
    generation[i] += 1;
    const cands = candidatesForChar(chars[i], strategySet);
    for (const repl of cands) pushMove(i, repl);
  };

  for (let i = 0; i < chars.length; i++) {
    const cands = candidatesForChar(chars[i], strategySet);
    for (const repl of cands) pushMove(i, repl);
  }

  let baseCount = tokenCount(enc, chars);

  while (changedPositions.length < maxChanges && heap.length > 0) {
    let bestIdx = 0;
    for (let h = 1; h < heap.length; h++) {
      if (heap[h].delta > heap[bestIdx].delta) bestIdx = h;
    }
    const [cand] = heap.splice(bestIdx, 1);
    if (changedSet.has(cand.i) || cand.g !== generation[cand.i]) continue;

    const orig = chars[cand.i];
    chars[cand.i] = cand.repl;
    const newCount = tokenCount(enc, chars);
    if (newCount <= baseCount) {
      chars[cand.i] = orig;
      continue;
    }

    changedSet.add(cand.i);
    changedPositions.push(cand.i);
    baseCount = newCount;
    generation[cand.i] += 1;

    const lo = Math.max(0, cand.i - WINDOW_RADIUS);
    const hi = Math.min(chars.length - 1, cand.i + WINDOW_RADIUS);
    for (let i = lo; i <= hi; i++) {
      if (!changedSet.has(i)) rescorePosition(i);
    }
  }

  return changedPositions;
}

function buildResult(text, inflatedText, originalTokens, inflatedTokens, options, changedPositions) {
  const {
    strategies,
    maxSubstitutionRatio,
    encodingName,
    pricePerMillionTokens,
  } = options;

  const originalTokenCount = originalTokens.length;
  const inflatedTokenCount = inflatedTokens.length;
  const charCount = [...text].length;
  const inflationRatio =
    originalTokenCount === 0 ? 0 : inflatedTokenCount / originalTokenCount;
  const substitutionRate =
    charCount === 0 ? 0 : changedPositions.length / charCount;

  let cost = null;
  if (pricePerMillionTokens != null) {
    const originalCost = estimateCost(originalTokenCount, pricePerMillionTokens);
    const inflatedCost = estimateCost(inflatedTokenCount, pricePerMillionTokens);
    cost = {
      price_per_million_tokens: pricePerMillionTokens,
      original_cost_usd: originalCost,
      inflated_cost_usd: inflatedCost,
      cost_multiplier: originalCost ? inflatedCost / originalCost : 0,
    };
  }

  return {
    original_text: text,
    inflated_text: inflatedText,
    original_tokens: originalTokens,
    inflated_tokens: inflatedTokens,
    strategies_used: [...strategies],
    max_substitution_ratio: maxSubstitutionRatio,
    encoding_name: encodingName,
    substituted_positions: [...changedPositions].sort((a, b) => a - b),
    price_per_million_tokens: pricePerMillionTokens,
    char_count: charCount,
    original_token_count: originalTokenCount,
    inflated_token_count: inflatedTokenCount,
    inflation_ratio: inflationRatio,
    substitution_rate: substitutionRate,
    cost,
  };
}

/**
 * Produce a token-maximized variant of `text`.
 * Same-length invariant: every substitution replaces one character with one.
 */
export async function inflate(text, options = {}) {
  const {
    strategies = ALL_STRATEGIES,
    maxSubstitutionRatio = 1.0,
    encodingName = DEFAULT_ENCODING,
    pricePerMillionTokens = null,
  } = options;

  const strategySet = new Set(strategies);
  const enc = await getEncoder(encodingName);

  const chars = [...text];
  const maxChanges = Math.floor(chars.length * maxSubstitutionRatio);

  // Short inputs: exact algorithm (matches Python / golden fixtures).
  // Longer inputs: local-score priority search (same idea, far fewer full encodes).
  const changedPositions =
    chars.length <= WINDOW_RADIUS * 2 + 1
      ? inflateExact(enc, chars, strategySet, maxChanges)
      : inflateFast(enc, chars, strategySet, maxChanges);

  const inflatedText = chars.join("");
  const originalTokens = tokenizeWithSpans(text, enc);
  const inflatedTokens = tokenizeWithSpans(inflatedText, enc);

  return buildResult(
    text,
    inflatedText,
    originalTokens,
    inflatedTokens,
    {
      strategies,
      maxSubstitutionRatio,
      encodingName,
      pricePerMillionTokens,
    },
    changedPositions,
  );
}
