import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { inflate } from "../src/inflator.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixtures = JSON.parse(
  readFileSync(join(__dirname, "../reference/golden_fixtures.json"), "utf8"),
);

describe("against golden_fixtures", () => {
  for (const fixture of fixtures) {
    const label = `${JSON.stringify(fixture.input)} [${fixture.strategies.join(",")}]`;
    it(label, async () => {
      const result = await inflate(fixture.input, {
        strategies: fixture.strategies,
        maxSubstitutionRatio: fixture.max_substitution_ratio,
        encodingName: fixture.encoding,
      });
      expect(result.inflated_text).toBe(fixture.expected_inflated_text);
      expect(result.original_token_count).toBe(fixture.expected_original_token_count);
      expect(result.inflated_token_count).toBe(fixture.expected_inflated_token_count);
      expect(result.char_count).toBe(fixture.expected_char_count);
      expect([...result.inflated_text].length).toBe([...result.original_text].length);
    });
  }
});
