import { MAX_INPUT_CHARS } from "./inflator.js";
import { utf8ByteCount } from "./tokenizer.js";

const inputEl = document.getElementById("input-text");
const inputHighlights = document.getElementById("input-highlights");
const inflatedView = document.getElementById("inflated-view");
const aggressivenessEl = document.getElementById("aggressiveness");
const aggressivenessValue = document.getElementById("aggressiveness-value");
const encodingEl = document.getElementById("encoding");
const copyBtn = document.getElementById("copy-btn");
const strategyInputs = [...document.querySelectorAll('input[name="strategy"]')];
const charCountEl = document.getElementById("char-count");
const inputStatusEl = document.getElementById("input-status");

const statInBytes = document.getElementById("stat-in-bytes");
const statOutBytes = document.getElementById("stat-out-bytes");
const statInTokens = document.getElementById("stat-in-tokens");
const statOutTokens = document.getElementById("stat-out-tokens");
const statCostRatio = document.getElementById("stat-cost-ratio");

inputEl.maxLength = MAX_INPUT_CHARS;

/** Cleared once on first focus so the demo seed doesn't need manual deleting. */
const demoSeedText = inputEl.value;
let demoSeedCleared = false;

let latestInflated = "";
let debounceTimer = null;
let inflateGeneration = 0;
let workerRequestId = 0;

const worker = new Worker(new URL("./inflate.worker.js", import.meta.url), {
  type: "module",
});

const pending = new Map();

worker.onmessage = (event) => {
  const { id, ok, result, error, name } = event.data;
  const entry = pending.get(id);
  if (!entry) return;
  pending.delete(id);
  if (ok) entry.resolve(result);
  else {
    const err = new Error(error ?? "Worker inflate failed");
    err.name = name ?? "Error";
    entry.reject(err);
  }
};

worker.onerror = (event) => {
  console.error("Inflate worker error", event);
};

function inflateInWorker(text, options) {
  const id = ++workerRequestId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, text, options });
  });
}

function renderColorized(container, tokens) {
  container.replaceChildren();
  const frag = document.createDocumentFragment();
  for (const t of tokens) {
    const span = document.createElement("span");
    span.className = `token token-${t.color_index}`;
    span.title = `id ${t.id}`;
    span.textContent = t.text;
    frag.appendChild(span);
  }
  frag.appendChild(document.createTextNode("\n"));
  container.appendChild(frag);
}

function renderPlainHighlights(text) {
  inputHighlights.replaceChildren();
  inputHighlights.appendChild(document.createTextNode(`${text}\n`));
}

function syncHighlightScroll() {
  inputHighlights.scrollTop = inputEl.scrollTop;
  inputHighlights.scrollLeft = inputEl.scrollLeft;
}

function selectedStrategies() {
  return strategyInputs.filter((el) => el.checked).map((el) => el.value);
}

function clampInput() {
  const chars = [...inputEl.value];
  if (chars.length > MAX_INPUT_CHARS) {
    inputEl.value = chars.slice(0, MAX_INPUT_CHARS).join("");
    return true;
  }
  return false;
}

function updateCharCount(truncated = false) {
  const n = [...inputEl.value].length;
  charCountEl.textContent = `${n} / ${MAX_INPUT_CHARS}`;
  charCountEl.classList.toggle("at-limit", n >= MAX_INPUT_CHARS);
  if (truncated) {
    inputStatusEl.hidden = false;
    inputStatusEl.textContent = `Input capped at ${MAX_INPUT_CHARS} characters.`;
  } else if (!inputStatusEl.textContent.startsWith("Computing")) {
    inputStatusEl.hidden = true;
    inputStatusEl.textContent = "";
  }
}

function setBusy(busy) {
  document.body.classList.toggle("is-busy", busy);
  if (busy) {
    inputStatusEl.hidden = false;
    inputStatusEl.textContent = "Computing…";
  } else if (inputStatusEl.textContent === "Computing…") {
    inputStatusEl.hidden = true;
    inputStatusEl.textContent = "";
  }
}

async function runInflate() {
  const generation = ++inflateGeneration;
  const truncated = clampInput();
  updateCharCount(truncated);

  const text = inputEl.value;
  const strategies = selectedStrategies();
  const maxSubstitutionRatio = Number(aggressivenessEl.value) / 100;
  const encodingName = encodingEl.value;

  aggressivenessValue.textContent = `${aggressivenessEl.value}%`;
  setBusy(true);

  try {
    const result = await inflateInWorker(text, {
      strategies,
      maxSubstitutionRatio,
      encodingName,
    });

    if (generation !== inflateGeneration) return;

    latestInflated = result.inflated_text;
    renderColorized(inputHighlights, result.original_tokens);
    renderColorized(inflatedView, result.inflated_tokens);
    syncHighlightScroll();

    statInBytes.textContent = String(utf8ByteCount(text));
    statOutBytes.textContent = String(utf8ByteCount(result.inflated_text));
    statInTokens.textContent = String(result.original_token_count);
    statOutTokens.textContent = String(result.inflated_token_count);

    if (result.original_token_count === 0) {
      statCostRatio.textContent = "—";
    } else {
      statCostRatio.textContent = `${result.inflation_ratio.toFixed(2)}×`;
    }
  } catch (err) {
    if (generation !== inflateGeneration) return;
    console.error(err);
    inputStatusEl.hidden = false;
    inputStatusEl.textContent = "Inflate failed — see console.";
  } finally {
    if (generation === inflateGeneration) setBusy(false);
  }
}

function scheduleInflate() {
  clearTimeout(debounceTimer);
  const len = [...inputEl.value].length;
  const delay = len > 200 ? 350 : 180;
  debounceTimer = setTimeout(() => {
    void runInflate();
  }, delay);
}

inputEl.addEventListener("focus", () => {
  if (demoSeedCleared) return;
  if (inputEl.value !== demoSeedText) {
    demoSeedCleared = true;
    return;
  }
  demoSeedCleared = true;
  inputEl.value = "";
  updateCharCount();
  renderPlainHighlights("");
  scheduleInflate();
});
inputEl.addEventListener("input", () => {
  const truncated = clampInput();
  updateCharCount(truncated);
  renderPlainHighlights(inputEl.value);
  syncHighlightScroll();
  scheduleInflate();
});
inputEl.addEventListener("scroll", syncHighlightScroll);
inputEl.addEventListener("paste", () => {
  requestAnimationFrame(() => {
    const truncated = clampInput();
    updateCharCount(truncated);
    renderPlainHighlights(inputEl.value);
  });
});

aggressivenessEl.addEventListener("input", () => {
  aggressivenessValue.textContent = `${aggressivenessEl.value}%`;
  scheduleInflate();
});
encodingEl.addEventListener("change", scheduleInflate);
for (const el of strategyInputs) {
  el.addEventListener("change", scheduleInflate);
}

copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(latestInflated);
    copyBtn.textContent = "copied";
    copyBtn.classList.add("copied");
    setTimeout(() => {
      copyBtn.textContent = "copy";
      copyBtn.classList.remove("copied");
    }, 1200);
  } catch {
    copyBtn.textContent = "fail";
  }
});

updateCharCount();
renderPlainHighlights(inputEl.value);
runInflate();
