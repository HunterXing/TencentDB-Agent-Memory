import { describe, expect, it } from "vitest";
import {
  buildSemanticRewritePrompt,
  compressJevStateSemantically,
  JEV_SEMANTIC_MAX_TOKENS_DEFAULT,
  measureJevState,
  normalizeRewrittenBody,
  resolveJevSemanticChunkChars,
  resolveJevSemanticMaxChunkFailures,
  resolveJevSemanticMaxDurationMs,
  resolveJevSemanticMaxScenes,
  resolveJevSemanticMaxTokens,
  resolveJevSemanticMinRatio,
  resolveJevSemanticTimeoutMs,
  splitSceneBody,
  validateRewrittenBody,
  estimateJevSemanticFeasibility,
  type SemanticRewriteFn,
} from "./jev-state-semantic.js";
import type { JevStatePayload } from "./jev-state-trim.js";

function makePayload(bodies: Record<string, string>): JevStatePayload {
  return {
    current_persona: "persona body",
    changed_scenes: Object.entries(bodies).map(([filename, content]) => ({
      filename,
      summary: `summary of ${filename}`,
      updated: "2026-10-19T00:00:00.000Z",
      content,
    })),
  };
}

describe("env resolution", () => {
  it("falls back to defaults when unset", () => {
    expect(resolveJevSemanticMaxTokens({} as NodeJS.ProcessEnv)).toBe(
      JEV_SEMANTIC_MAX_TOKENS_DEFAULT,
    );
    expect(resolveJevSemanticChunkChars({} as NodeJS.ProcessEnv)).toBe(3_000);
    expect(resolveJevSemanticTimeoutMs({} as NodeJS.ProcessEnv)).toBe(120_000);
    expect(resolveJevSemanticMinRatio({} as NodeJS.ProcessEnv)).toBe(0.2);
    expect(resolveJevSemanticMaxScenes({} as NodeJS.ProcessEnv)).toBe(4);
  });

  it("reads overrides from env", () => {
    const env = {
      JEV_SEMANTIC_MAX_TOKENS: "8000",
      JEV_SEMANTIC_TIMEOUT_MS: "5000",
      JEV_SEMANTIC_MIN_RATIO: "0.5",
      JEV_SEMANTIC_MAX_SCENES: "2",
      JEV_SEMANTIC_MAX_DURATION_MS: "60000",
      JEV_SEMANTIC_MAX_CHUNK_FAILURES: "5",
    } as NodeJS.ProcessEnv;
    expect(resolveJevSemanticMaxTokens(env)).toBe(8000);
    expect(resolveJevSemanticTimeoutMs(env)).toBe(5000);
    expect(resolveJevSemanticMinRatio(env)).toBe(0.5);
    expect(resolveJevSemanticMaxScenes(env)).toBe(2);
    expect(resolveJevSemanticMaxDurationMs(env)).toBe(60_000);
    expect(resolveJevSemanticMaxChunkFailures(env)).toBe(5);
  });

  it("falls back to the duration and failure defaults when unset", () => {
    expect(resolveJevSemanticMaxDurationMs({} as NodeJS.ProcessEnv)).toBe(180_000);
    expect(resolveJevSemanticMaxChunkFailures({} as NodeJS.ProcessEnv)).toBe(3);
  });

  it("rejects a malformed duration and failure count", () => {
    expect(resolveJevSemanticMaxDurationMs({ JEV_SEMANTIC_MAX_DURATION_MS: "abc" } as NodeJS.ProcessEnv)).toBe(180_000);
    expect(resolveJevSemanticMaxDurationMs({ JEV_SEMANTIC_MAX_DURATION_MS: "-5" } as NodeJS.ProcessEnv)).toBe(180_000);
    expect(resolveJevSemanticMaxDurationMs({ JEV_SEMANTIC_MAX_DURATION_MS: "0" } as NodeJS.ProcessEnv)).toBe(0);
    expect(resolveJevSemanticMaxChunkFailures({ JEV_SEMANTIC_MAX_CHUNK_FAILURES: "abc" } as NodeJS.ProcessEnv)).toBe(3);
    expect(resolveJevSemanticMaxChunkFailures({ JEV_SEMANTIC_MAX_CHUNK_FAILURES: "0" } as NodeJS.ProcessEnv)).toBe(3);
  });

  it("rejects malformed ratios instead of accepting them", () => {
    expect(resolveJevSemanticMinRatio({ JEV_SEMANTIC_MIN_RATIO: "0" } as NodeJS.ProcessEnv)).toBe(0.2);
    expect(resolveJevSemanticMinRatio({ JEV_SEMANTIC_MIN_RATIO: "1" } as NodeJS.ProcessEnv)).toBe(0.2);
    expect(resolveJevSemanticMinRatio({ JEV_SEMANTIC_MIN_RATIO: "abc" } as NodeJS.ProcessEnv)).toBe(0.2);
  });
});

describe("estimateJevSemanticFeasibility", () => {
  it("calls a single oversized scene feasible", () => {
    // One scene of 20k characters compressed to the 0.5 target fits easily.
    const payload = makePayload({ "a.md": "A".repeat(20_000) });
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    expect(estimate.feasible).toBe(true);
    expect(estimate.bestCaseFinalChars).toBeLessThan(32_000);
  });

  it("calls a payload that cannot fit even at the model's limit infeasible", () => {
    // Twelve scenes totalling 216k characters: even the lower bound a rewrite
    // realistically reaches does not fit, so the pass would burn an hour and
    // still fall back to the trim policy.
    const bodies: Record<string, string> = {};
    for (let i = 0; i < 12; i += 1) bodies[`s${i}.md`] = "X".repeat(18_000);
    const payload = makePayload(bodies);
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    expect(estimate.feasible).toBe(false);
    expect(estimate.worstCaseFinalChars).toBeGreaterThan(32_000);
  });

  it("rejects a payload that cannot fit at the target ratio", () => {
    // The target the prompt asks for is what the pass is judged on: a
    // payload that misses the limit at that ratio cannot reach it, and
    // starting the pass would only burn the time budget.
    const bodies: Record<string, string> = {};
    for (let i = 0; i < 4; i += 1) bodies[`s${i}.md`] = "X".repeat(18_000);
    const payload = makePayload(bodies);
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    // The best case does not fit ...
    expect(estimate.bestCaseFinalChars).toBeGreaterThan(32_000);
    // ... so the pass is not worth running.
    expect(estimate.feasible).toBe(false);
  });

  it("keeps a payload that fits at the target ratio feasible", () => {
    // Between the two bounds the outcome depends on how well the model
    // compresses — that is what the pass itself is for, so the pass runs.
    const bodies: Record<string, string> = {};
    for (let i = 0; i < 3; i += 1) bodies[`s${i}.md`] = "X".repeat(18_000);
    const payload = makePayload(bodies);
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    expect(estimate.bestCaseFinalChars).toBeLessThan(32_000);
    expect(estimate.feasible).toBe(true);
  });

  it("keeps the fixed part of the payload out of the ratio", () => {
    const payload = makePayload({ "a.md": "A".repeat(10_000) });
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    // The persona and the JSON scaffolding survive the compression, so the
    // fixed part is always added back on top of the compressed scenes.
    expect(estimate.fixedChars).toBeGreaterThan(0);
    expect(estimate.bestCaseFinalChars).toBe(estimate.fixedChars + 5_000);
  });

  it("reports a changed-scene set of zero characters as feasible", () => {
    const payload = makePayload({ "a.md": "" });
    const estimate = estimateJevSemanticFeasibility(payload, 32_000, 0.5);
    expect(estimate.sceneChars).toBe(0);
    expect(estimate.feasible).toBe(true);
  });
});

describe("normalizeRewrittenBody", () => {
  it("strips a markdown fence", () => {
    const out = normalizeRewrittenBody("```markdown\n## 章节\n内容。\n```");
    expect(out).toBe("## 章节\n内容。");
  });

  it("drops a prose preamble before the first heading", () => {
    const out = normalizeRewrittenBody("好的，这是压缩后的版本：\n## 章节\n内容。");
    expect(out.startsWith("## 章节")).toBe(true);
  });

  it("keeps a body that starts with a heading untouched", () => {
    const out = normalizeRewrittenBody("## 章节\n内容。");
    expect(out).toBe("## 章节\n内容。");
  });

  it("returns empty for an empty answer", () => {
    expect(normalizeRewrittenBody("   ")).toBe("");
  });
});

describe("validateRewrittenBody", () => {
  const original = "x".repeat(100);

  it("rejects an empty body", () => {
    expect(validateRewrittenBody("", original, 0.2)).toEqual({ ok: false, reason: "empty_output" });
  });

  it("rejects a body that is too small to be a faithful rewrite", () => {
    const out = validateRewrittenBody("短。", original, 0.2);
    expect(out).toEqual({ ok: false, reason: "too_small" });
  });

  it("rejects a body that did not shrink", () => {
    const out = validateRewrittenBody(original, original, 0.2);
    expect(out).toEqual({ ok: false, reason: "not_shrunk" });
  });

  it("rejects a body that looks cut mid-sentence", () => {
    const out = validateRewrittenBody("## 标题\n" + "文字".repeat(20) + "被截断的文字", original, 0.2);
    expect(out).toEqual({ ok: false, reason: "possibly_truncated" });
  });

  it("rejects a body that ends on a separator", () => {
    // A trailing comma or dash means the text was cut between two parts of
    // a sentence.
    expect(validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "，", original, 0.2))
      .toEqual({ ok: false, reason: "possibly_truncated" });
    expect(validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + " —", original, 0.2))
      .toEqual({ ok: false, reason: "possibly_truncated" });
  });

  it("rejects a body that was cut inside a code block", () => {
    const out = validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "\n```bash\nnpm run build", original, 0.2);
    expect(out).toEqual({ ok: false, reason: "possibly_truncated" });
  });

  it("accepts a body that ends with a table row", () => {
    // A technical document routinely ends with a table row. Requiring a
    // sentence-ending mark there threw away good rewrites.
    const out = validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "\n| 字段 | 说明 |", original, 0.2);
    expect(out).toEqual({ ok: true });
  });

  it("accepts a body that ends with a closing bracket or a quote", () => {
    expect(validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "（括号结束）", original, 0.2))
      .toEqual({ ok: true });
    expect(validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "“引用结束”", original, 0.2))
      .toEqual({ ok: true });
  });

  it("accepts a body that ends with a closed code block", () => {
    const out = validateRewrittenBody("## 标题\n" + "内容。".repeat(10) + "\n```bash\nnpm run build\n```", original, 0.2);
    expect(out).toEqual({ ok: true });
  });

  it("accepts a plausible rewrite", () => {
    const out = validateRewrittenBody("## 标题\n" + "内容。".repeat(10), original, 0.2);
    expect(out).toEqual({ ok: true });
  });
});

describe("buildSemanticRewritePrompt", () => {
  it("states the constraints and includes the original body", () => {
    const prompt = buildSemanticRewritePrompt("scene.md", "BODY-CONTENT");
    expect(prompt).toContain("scene.md");
    expect(prompt).toContain("BODY-CONTENT");
    expect(prompt).toContain("不要发明原文没有的事实");
    expect(prompt).toContain("只有改写后的正文");
  });
});

describe("splitSceneBody", () => {
  it("keeps a small body as a single chunk", () => {
    expect(splitSceneBody("## 标题\n正文内容。", 3_000)).toEqual(["## 标题\n正文内容。"]);
  });

  it("splits on markdown headings and keeps every heading", () => {
    const body = ["## 甲\n" + "甲".repeat(400), "## 乙\n" + "乙".repeat(400)].join("\n");
    const chunks = splitSceneBody(body, 500);
    expect(chunks.length).toBe(2);
    expect(chunks[0].startsWith("## 甲")).toBe(true);
    expect(chunks[1].startsWith("## 乙")).toBe(true);
  });

  it("cuts an oversized section on blank lines and repeats its heading", () => {
    const para = "内容。".repeat(200);
    const body = `## 长节\n\n${para}\n\n${para}`;
    const chunks = splitSceneBody(body, 600);
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.length).toBeLessThanOrEqual(600);
    }
    expect(chunks[0].startsWith("## 长节")).toBe(true);
  });

  it("never returns an empty chunk for an empty body", () => {
    expect(splitSceneBody("   ", 500)).toEqual([]);
  });

  it("cuts a section whose paragraphs are longer than the limit", () => {
    // A paragraph that is longer than the limit on its own cannot be cut on
    // blank lines, so the chunk came out at several times the limit. The
    // upstream gateway answers such a request with a 504 and the whole scene
    // is lost. On the production profile this produced chunks of 11556
    // characters against a limit of 3000.
    const hugeParagraph = "长".repeat(12_000);
    const body = `## 长节\n\n${hugeParagraph}`;
    const chunks = splitSceneBody(body, 3_000);
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.length).toBeLessThanOrEqual(3_000);
    }
  });

  it("keeps every chunk within the limit for a body of long lines", () => {
    // Lines without a blank line between them are the other case the blank
    // line split cannot handle.
    const body = ["## 标题", ...Array.from({ length: 300 }, () => "行".repeat(100))].join("\n");
    const chunks = splitSceneBody(body, 3_000);
    for (const chunk of chunks) {
      expect(chunk.length).toBeLessThanOrEqual(3_000);
    }
  });

  it("does not lose the text of a body that is cut", () => {
    // The cut must not silently drop content: every character of the body
    // has to survive in one of the chunks.
    const body = `## 长节\n\n${"长".repeat(12_000)}`;
    const chunks = splitSceneBody(body, 3_000);
    const rejoined = chunks.join("").replace(/\s+/g, "");
    const original = body.replace(/\s+/g, "");
    expect(rejoined.length).toBe(original.length);
  });
});

describe("compressJevStateSemantically", () => {
  // A realistic rewrite keeps roughly half of the original body.
  const rewriteBody = (size: number) => "## 压缩\n" + "内容。".repeat(Math.max(1, Math.floor(size / 3)));

  it("returns the payload untouched when it already fits", async () => {
    const payload = makePayload({ "a.md": "small" });
    const rewrite: SemanticRewriteFn = async () => "never";
    const outcome = await compressJevStateSemantically(payload, 100_000, rewrite);
    expect(outcome.result).not.toBeNull();
    expect(outcome.compressedScenes).toEqual([]);
    expect(outcome.result!.savedChars).toBe(0);
  });

  it("compresses the largest scene and keeps metadata", async () => {
    const payload = makePayload({
      "big.md": "A".repeat(5000),
      "small.md": "B".repeat(100),
    });
    const rewrite: SemanticRewriteFn = async ({ content }) => rewriteBody(content.length / 2);
    const outcome = await compressJevStateSemantically(payload, 3000, rewrite);
    expect(outcome.result).not.toBeNull();
    expect(outcome.compressedScenes).toContain("big.md");
    expect(outcome.result!.finalChars).toBeLessThanOrEqual(3000);
    const parsed = JSON.parse(outcome.result!.state);
    expect(parsed.changed_scenes[0].filename).toBe("big.md");
    expect(parsed.changed_scenes[0].summary).toBe("summary of big.md");
    expect(parsed.changed_scenes[0].updated).toBe("2026-10-19T00:00:00.000Z");
    expect(parsed.current_persona).toBe("persona body");
  });

  it("records a failure and keeps the original scene when the model throws", async () => {
    const payload = makePayload({ "big.md": "A".repeat(5000) });
    const rewrite: SemanticRewriteFn = async () => {
      throw new Error("provider down");
    };
    const outcome = await compressJevStateSemantically(payload, 3000, rewrite);
    expect(outcome.compressedScenes).toEqual([]);
    expect(outcome.skippedScenes).toEqual(["big.md"]);
    expect(outcome.failures[0].reason).toBe("chunk_error:provider down");
    // The scene body is untouched, so the payload still does not fit.
    expect(outcome.result).toBeNull();
  });

  it("rejects an implausible rewrite and keeps the original", async () => {
    const payload = makePayload({ "big.md": "A".repeat(5000) });
    const rewrite: SemanticRewriteFn = async () => "太短";
    const outcome = await compressJevStateSemantically(payload, 3000, rewrite);
    expect(outcome.compressedScenes).toEqual([]);
    expect(outcome.failures[0].reason).toBe("chunk_too_small");
  });

  it("keeps the original text of a chunk the model could not rewrite", async () => {
    // Two chunks: the first one is rewritten, the second one fails. The scene
    // still uses the rewritten first chunk and the original second chunk.
    const body = ["## 甲\n" + "甲".repeat(300), "## 乙\n" + "乙".repeat(300)].join("\n");
    const payload = makePayload({ "big.md": body });
    const rewrite: SemanticRewriteFn = async (input) => {
      if (input.content.includes("乙")) throw new Error("chunk down");
      return "## 甲\n" + "保留。".repeat(50);
    };
    const outcome = await compressJevStateSemantically(payload, 700, rewrite, { chunkChars: 400 });
    expect(outcome.compressedScenes).toEqual(["big.md"]);
    const state = JSON.parse(outcome.result?.state ?? "{}");
    const content = state.changed_scenes?.[0]?.content ?? "";
    expect(content).toContain("保留。");
    expect(content).toContain("乙");
  });

  it("returns a null result when the payload still does not fit", async () => {
    const payload = makePayload({ "big.md": "A".repeat(5000) });
    const rewrite: SemanticRewriteFn = async () => rewriteBody(2500);
    const outcome = await compressJevStateSemantically(payload, 10, rewrite);
    expect(outcome.result).toBeNull();
  });

  it("honours the max-scenes cap", async () => {
    const scene = (n: number) => "## 节\n\n" + "甲".repeat(n);
    const payload = makePayload({
      "a.md": scene(5000),
      "b.md": scene(5000),
      "c.md": scene(5000),
    });
    let calls = 0;
    const rewrite: SemanticRewriteFn = async ({ content }) => {
      calls += 1;
      return rewriteBody(content.length / 2);
    };
    await compressJevStateSemantically(payload, 3000, rewrite, { maxScenes: 2 });
    // Two scenes were attempted. The third one is never touched.
    expect(calls).toBeGreaterThan(0);
    expect(calls).toBeLessThan(8);
  });

  it("counts a failed scene against the max-scenes cap", async () => {
    // A scene whose rewrite keeps failing must not let the pass walk through
    // the whole library: the cap limits *attempts*, not successes. Otherwise a
    // provider that is down turns a four-scene budget into a twelve-scene
    // (103-request) one.
    //
    // The bodies are cut into a fixed number of chunks (one blank line per
    // paragraph) so the number of rewrite calls is the number of scenes.
    const scene = (n: number) => "## 节\n\n" + "甲".repeat(n);
    const payload = makePayload({
      "a.md": scene(2000),
      "b.md": scene(2000),
      "c.md": scene(2000),
    });
    let calls = 0;
    const rewrite: SemanticRewriteFn = async ({ content }) => {
      calls += 1;
      throw new Error("provider down");
    };
    const outcome = await compressJevStateSemantically(payload, 3000, rewrite, {
      maxScenes: 2,
      maxChunkFailures: 1,
    });
    // Two scenes were attempted, one chunk each — not three.
    expect(calls).toBe(2);
    expect(outcome.compressedScenes).toEqual([]);
    expect(outcome.skippedScenes).toHaveLength(2);
  });

  it("abandons a scene after too many consecutive chunk failures", async () => {
    // A scene whose provider is unreachable must not pay for every one of its
    // chunks: the pass gives up after the configured number of failures in a
    // row and keeps the remaining chunks as they are.
    const body = ["## 甲\n" + "甲".repeat(300), "## 乙\n" + "乙".repeat(300), "## 丙\n" + "丙".repeat(300)].join("\n");
    const payload = makePayload({ "big.md": body });
    let calls = 0;
    const rewrite: SemanticRewriteFn = async () => {
      calls += 1;
      throw new Error("provider down");
    };
    const outcome = await compressJevStateSemantically(payload, 700, rewrite, {
      chunkChars: 400,
      maxChunkFailures: 2,
    });
    // Two failures in a row are enough to abandon the third chunk.
    expect(calls).toBe(2);
    expect(outcome.failures.some((f) => f.reason.includes("abandoned_after_2_consecutive_failures"))).toBe(true);
  });

  it("stops the pass when the wall-clock budget is exhausted", async () => {
    // A rewrite request costs 8-45s against the real gateway. Without a
    // budget the pass can occupy the L3 task for an hour and still defer.
    const payload = makePayload({
      "a.md": "A".repeat(5000),
      "b.md": "B".repeat(5000),
      "c.md": "C".repeat(5000),
    });
    let calls = 0;
    const rewrite: SemanticRewriteFn = async ({ content }) => {
      calls += 1;
      // Burn past the budget on the first call.
      await new Promise((resolve) => setTimeout(resolve, 40));
      return rewriteBody(content.length / 2);
    };
    const outcome = await compressJevStateSemantically(payload, 10, rewrite, { maxDurationMs: 10 });
    expect(outcome.failures.some((f) => f.reason === "max_duration_exceeded")).toBe(true);
    // The budget stopped the pass long before the third scene was attempted.
    expect(calls).toBeLessThan(3);
  });

  it("keeps the original text of the chunks the budget did not reach", async () => {
    // The first chunk is rewritten (the budget is still alive when it starts),
    // the budget runs out during that call, and the second chunk is kept as it
    // was. The scene is still compressed — one rewritten chunk is enough to
    // make the payload fit — and the untouched chunk keeps its text.
    const body = ["## 甲\n" + "甲".repeat(300), "## 乙\n" + "乙".repeat(300)].join("\n");
    const payload = makePayload({ "big.md": body });
    const rewrite: SemanticRewriteFn = async ({ content }) => {
      await new Promise((resolve) => setTimeout(resolve, 40));
      return "## 甲\n" + "保留。".repeat(50);
    };
    const outcome = await compressJevStateSemantically(payload, 700, rewrite, {
      chunkChars: 400,
      maxDurationMs: 10,
    });
    expect(outcome.result).not.toBeNull();
    const state = JSON.parse(outcome.result?.state ?? "{}");
    const content = state.changed_scenes?.[0]?.content ?? "";
    expect(content).toContain("保留。");
    // The second chunk was never rewritten, so its text is still there.
    expect(content).toContain("乙");
    expect(outcome.failures.some((f) => f.reason === "budget_exhausted")).toBe(true);
  });

  it("compresses the largest scene first", async () => {
    const payload = makePayload({
      "small.md": "S".repeat(100),
      "huge.md": "H".repeat(9000),
    });
    const seen: string[] = [];
    const rewrite: SemanticRewriteFn = async ({ filename, content }) => {
      seen.push(filename);
      return rewriteBody(content.length / 2);
    };
    await compressJevStateSemantically(payload, 3000, rewrite);
    expect(seen[0]).toBe("huge.md");
  });
});

describe("measureJevState", () => {
  it("matches JSON.stringify length", () => {
    const payload = makePayload({ "a.md": "abc" });
    expect(measureJevState(payload)).toBe(JSON.stringify(payload).length);
  });
});
