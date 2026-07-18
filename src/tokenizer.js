import { Tiktoken } from "js-tiktoken/lite";

export const DEFAULT_ENCODING = "cl100k_base";
export const PALETTE_SIZE = 8;

const encodingCache = new Map();
const rankLoaders = {
  cl100k_base: () => import("js-tiktoken/ranks/cl100k_base"),
  o200k_base: () => import("js-tiktoken/ranks/o200k_base"),
};

/** Encodings offered in the UI (GPT-4 / GPT-4o families). */
export const AVAILABLE_ENCODINGS = [
  { name: "cl100k_base", label: "GPT-4 family (cl100k_base)" },
  { name: "o200k_base", label: "GPT-4o / GPT-5 family (o200k_base)" },
];

/**
 * Load (and cache) a Tiktoken encoder. Async because rank tables are
 * code-split for smaller initial downloads on GitHub Pages.
 */
export async function getEncoder(encodingName = DEFAULT_ENCODING) {
  let enc = encodingCache.get(encodingName);
  if (!enc) {
    const loader = rankLoaders[encodingName];
    if (!loader) {
      throw new Error(`Unsupported encoding: ${encodingName}`);
    }
    const mod = await loader();
    enc = new Tiktoken(mod.default);
    encodingCache.set(encodingName, enc);
  }
  return enc;
}

/**
 * Byte length of a single token — js-tiktoken equivalent of
 * tiktoken's decode_single_token_bytes.
 */
export function decodeSingleTokenBytes(enc, tokenId) {
  const bytes = enc.textMap.get(tokenId) ?? enc.inverseSpecialTokens[tokenId];
  if (bytes == null) {
    throw new Error(`Unknown token id: ${tokenId}`);
  }
  return bytes;
}

/**
 * Tokenize `text` and return each token with its exact character span.
 * Uses raw token bytes (not string decode of a lone id) so multi-byte
 * UTF-8 characters that straddle token boundaries stay correct.
 */
export function tokenizeWithSpans(text, enc) {
  const tokenIds = enc.encode(text);
  // Code-point characters — matches Python's str indexing.
  const chars = [...text];

  const byteToChar = [];
  for (let i = 0; i < chars.length; i++) {
    const byteLen = new TextEncoder().encode(chars[i]).length;
    for (let b = 0; b < byteLen; b++) {
      byteToChar.push(i);
    }
  }
  byteToChar.push(chars.length);

  const tokens = [];
  let bytePos = 0;
  for (let idx = 0; idx < tokenIds.length; idx++) {
    const tid = tokenIds[idx];
    const raw = decodeSingleTokenBytes(enc, tid);
    const startChar = byteToChar[bytePos];
    const endByte = bytePos + raw.length;
    const endChar =
      endByte < byteToChar.length ? byteToChar[endByte] : chars.length;
    tokens.push({
      index: idx,
      id: tid,
      text: chars.slice(startChar, endChar).join(""),
      start: startChar,
      end: endChar,
      color_index: idx % PALETTE_SIZE,
    });
    bytePos = endByte;
  }
  return tokens;
}

export function utf8ByteCount(text) {
  return new TextEncoder().encode(text).length;
}
