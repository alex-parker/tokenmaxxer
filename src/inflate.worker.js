import { inflate } from "./inflator.js";

self.onmessage = async (event) => {
  const { id, text, options } = event.data;
  try {
    const result = await inflate(text, options);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err?.message ?? String(err),
      name: err?.name,
    });
  }
};
