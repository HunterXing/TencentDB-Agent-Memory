/**
 * Semantic JEV state overflow handling.
 *
 * `trim` shrinks an oversized JEV payload positionally: it chops the largest
 * scene in half and keeps the head and tail. The cut lands wherever the byte
 * budget runs out, so the paragraphs the decision most needs can be the ones
 * that disappear — and the decision then reasons over a mutilated scene.
 *
 * This module takes a different route: it asks a model to rewrite the
 * oversized scene into a shorter but semantically faithful version — stale or
 * superseded facts dropped, repeated narration collapsed, every still-valid
 * fact kept — and re-serializes the payload with the rewritten content.
 *
 * The result is lossy in *content* rather than in *position*: what survives is
 * still a true description of the scene, so the decision sees real
 * information instead of a chopped middle.
 *
 * The model call is injected as a plain function so this module stays testable
 * without a network, and so the caller can decide which runner/provider to
 * use. Everything here is side-effect free: nothing is written back to disk —
 * the gate only needs a smaller payload for the *current* decision.
 *
 * Env knobs (all optional; defaults preserve upstream behavior):
 *   JEV_SEMANTIC_MAX_TOKENS     max tokens for the rewrite call, default 4000
 *                              — see JEV_SEMANTIC_MAX_TOKENS_DEFAULT for why
 *                              this is kept small deliberately
 *   JEV_SEMANTIC_TIMEOUT_MS      timeout for the rewrite call, default 120000
 *   JEV_SEMANTIC_MIN_RATIO       minimum fraction of the original that must
 *                                survive, default 0.2 — guards against a
 *                                model that returns an empty or stub answer
 *   JEV_SEMANTIC_MAX_SCENES      how many scenes to attempt per pass,
 *                                default 4 — larger sets are handled by
 *                                repeated passes. A scene that fails still
 *                                counts as an attempt.
 *   JEV_SEMANTIC_CHUNK_CHARS     maximum characters per rewrite request,
 *                                default 3000 — the body is rewritten in
 *                                pieces of this size
 *   JEV_SEMANTIC_MAX_DURATION_MS wall-clock budget for the whole pass,
 *                                default 180000. A rewrite request costs
 *                                8-45s against a real gateway, so an
 *                                unbounded pass can spend an hour on a
 *                                large library and still not fit the limit.
 *                                When the budget runs out the remaining
 *                                chunks keep their original text.
 *   JEV_SEMANTIC_MAX_CHUNK_FAILURES  consecutive failed chunks after which a
 *                                scene is abandoned, default 3. Without it a
 *                                scene whose provider is down pays for every
 *                                one of its chunks before giving up.
 */

import type { JevStatePayload } from "./jev-state-trim.js";

/**
 * Max output tokens for one rewrite call.
 *
 * Kept deliberately small. Against a reasoning model this value — not the
 * input size — dominates latency: measured on 88-char inputs, a 16000 budget
 * averaged 43.9s per call (worst case 60079ms, 1 of 3 succeeding) while 4000
 * averaged 31.0s with no failures. Chunk size barely mattered by comparison
 * (800-char and 3000-char chunks both landed near 47s), so every extra token
 * of budget is just more thinking time spent walking into the upstream
 * gateway's response timeout. A tight budget also yields a shorter, less
 * padded rewrite, which is what this pass is for.
 *
 * Override with JEV_SEMANTIC_MAX_TOKENS when the rewrite model changes; a
 * non-reasoning model can afford a much larger budget.
 */
export const JEV_SEMANTIC_MAX_TOKENS_DEFAULT = 4_000;
export const JEV_SEMANTIC_TIMEOUT_MS_DEFAULT = 120_000;
export const JEV_SEMANTIC_MIN_RATIO_DEFAULT = 0.2;
export const JEV_SEMANTIC_MAX_SCENES_DEFAULT = 4;
export const JEV_SEMANTIC_CHUNK_CHARS_DEFAULT = 3_000;
/** Chunks below this size are kept as they are instead of being rewritten. */
export const JEV_SEMANTIC_MIN_CHUNK_CHARS_DEFAULT = 200;
/**
 * Wall-clock budget for one whole compression pass.
 *
 * A rewrite request against the production gateway costs 8-45s. A library
 * whose changed-scene set is large enough to overflow the limit therefore
 * needs dozens of requests, and an unbounded pass would keep the L3 task
 * busy for an hour while the gate still ends up deferring. The budget makes
 * the pass bounded: whatever has not been rewritten keeps its original text
 * and the caller falls back to the positional trim policy.
 */
export const JEV_SEMANTIC_MAX_DURATION_MS_DEFAULT = 180_000;
/**
 * Consecutive failed chunks after which a scene is abandoned.
 *
 * Without this a scene whose provider is unreachable pays for every one of
 * its chunks (a 37k-char scene is 12 requests) before it is reported as
 * failed.
 */
export const JEV_SEMANTIC_MAX_CHUNK_FAILURES_DEFAULT = 3;

/**
 * Lower bound on the ratio a rewrite can realistically reach.
 *
 * Measured against the production gateway a rewrite lands between 0.25 and
 * 0.55 of the original. A model asked for less than this returns something
 * that fails the plausibility check instead.
 */
/**
 * The smallest fraction of the original a rewrite can realistically reach.
 *
 * The rewrite prompt asks for 0.5, and a well-behaved model lands near it:
 * a measured production pass took a 36783-character scene body down to
 * about 23500 characters, a ratio of 0.64. Reaching much below that needs
 * the model to drop content, which the plausibility check then rejects, so
 * this is a floor rather than a promise.
 *
 * It is deliberately *not* used to decide whether a pass is worth starting.
 * That decision uses the target ratio the prompt actually asks for — see
 * estimateJevSemanticFeasibility.
 */
export const JEV_SEMANTIC_MIN_ACHIEVABLE_RATIO = 0.4;

/**
 * Estimate whether a semantic pass can bring the payload under the limit.
 *
 * Two numbers matter:
 *
 *   bestCaseFinalChars  — every scene shrunk to `targetRatio`. This is the
 *                          outcome the rewrite prompt asks for, so a pass that
 *                          lands here comfortably fits.
 *   worstCaseFinalChars — every scene shrunk to the lowest ratio a rewrite
 *                          realistically reaches. A pass that still does not
 *                          fit at this point cannot succeed at all, no
 *                          matter how long it is given.
 *
 * `feasible` asks whether the target the prompt asks for can fit under the
 * limit. Between the two numbers the outcome depends on how well the model
 * compresses, and that is what the pass itself is for — so a payload that
 * fits at the target is worth attempting even if a given run lands above it
 * (the pass then falls back to the trim policy). A payload that cannot fit
 * at the target is rejected outright, because such a pass would burn the
 * whole time budget and fall back to the trim policy anyway.
 *
 * Returns the estimated sizes and whether the pass is worth attempting.
 */
export function estimateJevSemanticFeasibility(
  payload: JevStatePayload,
  limit: number,
  targetRatio: number,
  opts: { minAchievableRatio?: number } = {},
): {
  feasible: boolean;
  bestCaseFinalChars: number;
  worstCaseFinalChars: number;
  sceneChars: number;
  fixedChars: number;
} {
  const minAchievable = opts.minAchievableRatio ?? JEV_SEMANTIC_MIN_ACHIEVABLE_RATIO;
  const sceneChars = payload.changed_scenes.reduce((sum, scene) => sum + scene.content.length, 0);
  // The fixed part of the payload: the persona plus the JSON scaffolding
  // around the scenes (filenames, summaries, timestamps).
  const fixedChars = measureJevState({ ...payload, changed_scenes: [] });
  // Best case: the model reaches the target ratio for every scene.
  const bestCaseFinalChars = Math.round(fixedChars + sceneChars * targetRatio);
  // Worst realistic case: the model only reaches its lower bound.
  const worstCaseFinalChars = Math.round(fixedChars + sceneChars * minAchievable);
  return {
    // Judge on the target the rewrite prompt actually asks for. A pass that
    // cannot hit the target cannot hit the limit either, and starting it
    // would spend the whole time budget to end in the trim fallback anyway.
    // The measured ratio settles it: a production pass reached 0.64 of the
    // original, so a 0.5 target is realistic and anything the target cannot
    // fit is out of reach.
    feasible: bestCaseFinalChars <= limit,
    bestCaseFinalChars,
    worstCaseFinalChars,
    sceneChars,
    fixedChars,
  };
}
/**
 * Target length of a rewritten chunk, as a fraction of the original.
 *
 * Without a target a model treats "shorter" as a light edit and returns
 * something barely smaller than the input, which does not free enough room
 * to matter.
 */
export const JEV_SEMANTIC_TARGET_RATIO_DEFAULT = 0.5;

export function resolveJevSemanticTargetRatio(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_TARGET_RATIO?.trim();
  if (raw && /^[0-9]*\.?[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0 && parsed < 1) return parsed;
  }
  return JEV_SEMANTIC_TARGET_RATIO_DEFAULT;
}

/**
 * Maximum size of one rewrite request, in characters.
 *
 * The chunk size is what keeps the rewrite usable in practice: a request with
 * a large body is either rejected by a gateway wall clock (504) or answered by
 * a reasoning model that spends the whole token budget thinking and returns no
 * body. Sections are rewritten in pieces of this size instead.
 */
export function resolveJevSemanticChunkChars(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_CHUNK_CHARS?.trim();
  if (raw && /^[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed >= 200) return parsed;
  }
  return JEV_SEMANTIC_CHUNK_CHARS_DEFAULT;
}

export function resolveJevSemanticMaxTokens(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_MAX_TOKENS?.trim();
  if (raw && /^[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0) return parsed;
  }
  return JEV_SEMANTIC_MAX_TOKENS_DEFAULT;
}

export function resolveJevSemanticTimeoutMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_TIMEOUT_MS?.trim();
  if (raw && /^[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0) return parsed;
  }
  return JEV_SEMANTIC_TIMEOUT_MS_DEFAULT;
}

export function resolveJevSemanticMinRatio(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_MIN_RATIO?.trim();
  if (!raw) return JEV_SEMANTIC_MIN_RATIO_DEFAULT;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed >= 1) return JEV_SEMANTIC_MIN_RATIO_DEFAULT;
  return parsed;
}

export function resolveJevSemanticMaxScenes(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_MAX_SCENES?.trim();
  if (raw && /^[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0) return parsed;
  }
  return JEV_SEMANTIC_MAX_SCENES_DEFAULT;
}

/**
 * Wall-clock budget for one whole compression pass.
 *
 * A negative or zero value disables the budget, which restores the
 * unbounded behaviour but is not the default.
 */
export function resolveJevSemanticMaxDurationMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_MAX_DURATION_MS?.trim();
  if (raw && /^-?[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0) return parsed;
    if (parsed === 0) return 0;
  }
  return JEV_SEMANTIC_MAX_DURATION_MS_DEFAULT;
}

/**
 * Consecutive failed chunks after which a scene is abandoned.
 */
export function resolveJevSemanticMaxChunkFailures(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.JEV_SEMANTIC_MAX_CHUNK_FAILURES?.trim();
  if (raw && /^[0-9]+$/.test(raw)) {
    const parsed = Number(raw);
    if (parsed > 0) return parsed;
  }
  return JEV_SEMANTIC_MAX_CHUNK_FAILURES_DEFAULT;
}

/**
 * A model call that rewrites one scene body. Returns the rewritten body, or
 * an empty string / null when the model produced nothing usable.
 */
export type SemanticRewriteFn = (input: {
  filename: string;
  content: string;
  maxTokens: number;
  timeoutMs: number;
}) => Promise<string | null | undefined>;

export interface JevSemanticResult {
  /** Re-serialized state payload after the semantic pass. */
  state: string;
  /** Serialized length before the semantic pass. */
  originalChars: number;
  /** Serialized length after the semantic pass. */
  finalChars: number;
  /** Chars saved by the semantic pass. */
  savedChars: number;
  /** Scenes whose body was rewritten. */
  compressedScenes: string[];
  /** Scenes left untouched (already small, or the rewrite failed). */
  skippedScenes: string[];
}

export interface JevSemanticOutcome {
  /**
   * The compressed payload, or `null` when the payload still does not fit —
   * the caller then falls back to the positional trim policy.
   */
  result: JevSemanticResult | null;
  /** Per-scene failure reasons, reported even when the pass did not succeed. */
  failures: Array<{ filename: string; reason: string }>;
  /** Scenes that were rewritten successfully. */
  compressedScenes: string[];
  /** Scenes left untouched (already small, or the rewrite failed). */
  skippedScenes: string[];
}

/** Serialized size of a payload, using the same JSON shape the gate sends. */
export function measureJevState(payload: JevStatePayload): number {
  return JSON.stringify(payload).length;
}

/**
 * Strip a model answer down to the rewritten body.
 *
 * Models wrap answers in markdown fences, prepend preambles, or append
 * trailing commentary. A preamble is not scene content and would pollute the
 * payload the decision reads, so it is removed. If the answer cannot be
 * recognized as content at all, the original is kept.
 */
export function normalizeRewrittenBody(raw: string): string {
  let text = raw.trim();
  if (!text) return "";
  // Remove a leading fenced block, keeping the content inside it.
  const fence = /^```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)\n?```\s*$/;
  const fenced = text.match(fence);
  if (fenced) text = fenced[1].trim();
  // Drop a leading "here is the compressed version" style preamble: a first
  // line that is prose rather than a markdown heading.
  const lines = text.split("\n");
  if (lines.length > 1 && !lines[0].trimStart().startsWith("#")) {
    const headingIdx = lines.findIndex((l) => l.trimStart().startsWith("#"));
    if (headingIdx > 0) text = lines.slice(headingIdx).join("\n").trim();
  }
  return text.trim();
}

/**
 * Check that a rewritten body is plausible: non-empty, big enough relative to
 * the original, and not obviously truncated mid-sentence.
 *
 * A model that hits its token limit returns `finish_reason: length` and the
 * body ends mid-word; that must be rejected rather than sent to the decision
 * provider, otherwise the decision reads a half-scene.
 */
export function validateRewrittenBody(
  rewritten: string,
  original: string,
  minRatio: number,
): { ok: true } | { ok: false; reason: string } {
  if (!rewritten) return { ok: false, reason: "empty_output" };
  if (rewritten.length < Math.floor(original.length * minRatio)) {
    return { ok: false, reason: "too_small" };
  }
  if (rewritten.length >= original.length) {
    return { ok: false, reason: "not_shrunk" };
  }
  // A body cut mid-sentence usually ends without terminal punctuation.
  // The set of endings that count as "finished" is wider than the usual
  // sentence punctuation: a Chinese technical document routinely ends with a
  // table row, a closed code block, a closing bracket or a quoted phrase,
  // and demanding a full stop there threw away perfectly good rewrites.
  //
  // What still marks a truncated body is the absence of any finishing mark
  // at all (the text was cut between two parts of a sentence) or a code fence
  // that was left open.
  const tail = rewritten.slice(-24).trim();
  if (tail.length > 0) {
    const finished = /[.!?。！？；;)\]}】」』"'`）｜|”’]$/.test(tail);
    if (!finished) {
      return { ok: false, reason: "possibly_truncated" };
    }
    // An odd number of code fences means the body was cut inside a code block.
    const fences = (rewritten.match(/```/g) ?? []).length;
    if (fences % 2 === 1) {
      return { ok: false, reason: "possibly_truncated" };
    }
  }
  return { ok: true };
}

/**
 * Split a scene body into independently rewritable chunks.
 *
 * A whole-scene rewrite is one large request, which two things break in
 * practice: a reasoning model spends its token budget thinking and returns no
 * body at all, and a gateway with a per-request wall clock answers 504 before
 * the model is done. Chunking by markdown section keeps every request small
 * enough for both, and it also lets a partially failed chunk stay as the
 * original text instead of losing the whole scene.
 *
 * Sections are split on markdown headings; a section that is still too long
 * is cut on blank lines, and the pieces keep the heading of the section they
 * belong to, so the reassembled body keeps its outline.
 */
export function splitSceneBody(body: string, maxChunkChars: number, minChunkChars = 200): string[] {
  const limit = Math.max(minChunkChars, Math.floor(maxChunkChars));
  const trimmed = body.trim();
  if (!trimmed) return [];
  if (trimmed.length <= limit) return [trimmed];

  // Split into markdown sections, keeping the heading with its content.
  const sections: string[] = [];
  let current = "";
  for (const line of trimmed.split("\n")) {
    const isHeading = /^#{1,6}\s/.test(line);
    if (isHeading && current.trim()) {
      sections.push(current.trim());
      current = "";
    }
    current += (current ? "\n" : "") + line;
  }
  if (current.trim()) sections.push(current.trim());

  const chunks: string[] = [];
  for (const section of sections) {
    if (section.length <= limit) {
      chunks.push(section);
      continue;
    }
    // Too long: cut on blank lines, keeping the heading on the first piece.
    const headingMatch = section.match(/^(#{1,6}\s.*)\n/);
    const heading = headingMatch ? headingMatch[1] : "";
    const rest = headingMatch ? section.slice(headingMatch[0].length) : section;
    let buffer = heading ? heading + "\n\n" : "";
    for (const para of rest.split(/\n{2,}/)) {
      const candidate = buffer ? `${buffer}\n\n${para}` : para;
      if (candidate.length > limit && buffer.trim()) {
        chunks.push(buffer.trim());
        buffer = para;
      } else {
        buffer = candidate;
      }
    }
    if (buffer.trim()) chunks.push(buffer.trim());
  }
  // A paragraph that is longer than the limit on its own (a long table, a
  // code block, a run of text without a blank line) cannot be cut on blank
  // lines, so the chunk above would come out far above the limit. The
  // request would then carry several times the intended payload, which the
  // upstream gateway answers with a 504 after its own timeout — the whole
  // rewrite of that scene is then lost.
  //
  // Every chunk is therefore cut down to the limit here, on blank lines
  // first, then on line breaks, then on raw characters as a last resort.
  const bounded: string[] = [];
  for (const chunk of chunks) {
    if (chunk.length <= limit) {
      bounded.push(chunk);
      continue;
    }
    bounded.push(...hardSplitChunk(chunk, limit));
  }
  return bounded.filter((c) => c.trim().length > 0);
}

/**
 * Cut a chunk that is longer than the limit into pieces of at most `limit`
 * characters, preferring blank lines, then line breaks, then raw characters.
 */
function hardSplitChunk(chunk: string, limit: number): string[] {
  const pieces: string[] = [];
  let rest = chunk;
  while (rest.length > limit) {
    // Prefer a blank line inside the limit.
    const window = rest.slice(0, limit);
    const blankIdx = window.lastIndexOf("\n\n");
    if (blankIdx > 0) {
      pieces.push(rest.slice(0, blankIdx).trim());
      rest = rest.slice(blankIdx).trimStart();
      continue;
    }
    // Otherwise a line break inside the limit.
    const lineIdx = window.lastIndexOf("\n");
    if (lineIdx > 0) {
      pieces.push(rest.slice(0, lineIdx).trim());
      rest = rest.slice(lineIdx).trimStart();
      continue;
    }
    // Last resort: cut on raw characters.
    pieces.push(rest.slice(0, limit).trim());
    rest = rest.slice(limit).trimStart();
  }
  if (rest.trim()) pieces.push(rest.trim());
  return pieces;
}

/**
 * Build the rewrite prompt. Kept separate so it can be asserted in tests.
 *
 * The body arrives one section at a time, so the prompt asks for a section
 * rather than a whole document, and it names a target length: without one, a
 * model treats "shorter" as a light edit and returns something barely smaller
 * than the input, which defeats the whole point of the pass.
 */
export function buildSemanticRewritePrompt(
  filename: string,
  content: string,
  opts: { targetRatio?: number } = {},
): string {
  const targetRatio = opts.targetRatio ?? 0.5;
  const targetChars = Math.max(200, Math.round(content.length * targetRatio));
  return [
    "你是一个记忆库整理器。下面是某个用户某个主题的记忆文档中的一个章节。",
    "",
    `任务：把这一节改写成**更短但信息不丢**的版本，目标长度约 ${targetChars} 字符（原文约 ${content.length} 字符）。`,
    "",
    "必须遵守：",
    "1. **删除过时/已被推翻的内容**：已被后续状态取代的旧事实、已完成的临时任务记录、重复的复述流水、只记热度的元记录。",
    "2. **保留当前仍然成立的**：硬事实、规则/铁律、当前架构与状态、偏好、未决的问题。",
    "3. **保留章节结构**：保留原文的 markdown 标题层级，不要凭空新增章节。",
    "4. **演变轨迹**类章节只留关键转变点（3-8 条），不要留流水账。",
    "5. **不要发明原文没有的事实**。",
    "6. 输出**只有改写后的正文**，不要任何前言、解释、代码块围栏。",
    "7. 结尾必须是完整句子，不要在句子中间截断。",
    "",
    `原文件：${filename}`,
    "",
    "原文：",
    "---",
    content,
  ].join("\n");
}

/**
 * Compress the payload until it fits `limit` by rewriting scene bodies.
 *
 * Scenes are compressed largest-first, because the largest bodies carry the
 * most redundancy. Every scene keeps its filename, summary and updated stamp;
 * only the body is replaced. When a rewrite fails or produces something
 * implausible, that scene is left as-is and the failure is recorded — a
 * partial result is still better than dropping the whole payload.
 *
 * Only the scene bodies are rewritten, not the persona. The persona is the
 * current state of the user, not a running log: rewriting it would risk
 * dropping a fact that is still true, and it is small anyway — on the
 * production profile it is 2075 characters, about 1% of the payload.
 *
 * Returns `null` only when the payload cannot be brought under the limit at
 * all; the caller then falls back to the positional trim policy.
 */
export async function compressJevStateSemantically(
  payload: JevStatePayload,
  limit: number,
  rewrite: SemanticRewriteFn,
  opts: {
    maxTokens?: number;
    timeoutMs?: number;
    minRatio?: number;
    maxScenes?: number;
    chunkChars?: number;
    minChunkChars?: number;
    maxDurationMs?: number;
    maxChunkFailures?: number;
    env?: NodeJS.ProcessEnv;
  } = {},
): Promise<JevSemanticOutcome> {
  const env = opts.env ?? process.env;
  const maxTokens = opts.maxTokens ?? resolveJevSemanticMaxTokens(env);
  const timeoutMs = opts.timeoutMs ?? resolveJevSemanticTimeoutMs(env);
  const minRatio = opts.minRatio ?? resolveJevSemanticMinRatio(env);
  const maxScenes = opts.maxScenes ?? resolveJevSemanticMaxScenes(env);
  const chunkChars = opts.chunkChars ?? resolveJevSemanticChunkChars(env);
  // A chunk below this size is kept as it is: rewriting it costs a request
  // and a wait, and it cannot free enough room to matter.
  const minChunkChars = opts.minChunkChars ?? JEV_SEMANTIC_MIN_CHUNK_CHARS_DEFAULT;
  const maxDurationMs = opts.maxDurationMs ?? resolveJevSemanticMaxDurationMs(env);
  const maxChunkFailures = opts.maxChunkFailures ?? resolveJevSemanticMaxChunkFailures(env);

  const original = measureJevState(payload);
  const scenes = payload.changed_scenes.map((scene) => ({ ...scene }));
  const working: JevStatePayload = { ...payload, changed_scenes: scenes };
  const compressed: string[] = [];
  const skipped: string[] = [];
  const failures: Array<{ filename: string; reason: string }> = [];
  // Wall-clock budget for the whole pass. Every rewrite request costs 8-45s
  // against a real gateway, so an unbounded pass over a large changed-scene
  // set keeps the L3 task busy for an hour and still ends up deferring. The
  // budget bounds the cost; whatever is left keeps its original text.
  const deadline = maxDurationMs > 0 ? Date.now() + maxDurationMs : Number.POSITIVE_INFINITY;
  let budgetExhausted = false;

  if (original <= limit) {
    return {
      result: {
        state: JSON.stringify(working),
        originalChars: original,
        finalChars: original,
        savedChars: 0,
        compressedScenes: [],
        skippedScenes: [],
      },
      failures,
      compressedScenes: [],
      skippedScenes: [],
    };
  }

  // Largest body first — it holds the most redundancy and buys the most room.
  const order = scenes
    .map((scene, index) => ({ index, len: scene.content.length }))
    .sort((a, b) => b.len - a.len)
    .map((entry) => entry.index);

  // Scenes are *attempted*, not merely compressed: a scene whose rewrite
  // failed still consumed a slot. Counting only the successful ones would let
  // a provider that is down push the pass through every scene in the library.
  let attempts = 0;

  for (const index of order) {
    if (measureJevState(working) <= limit) break;
    if (attempts >= maxScenes) break;
    if (Date.now() >= deadline) {
      budgetExhausted = true;
      failures.push({ filename: scenes[index].filename, reason: "budget_exhausted" });
      break;
    }
    const scene = scenes[index];
    if (!scene.content || scene.content.length === 0) continue;
    attempts += 1;

    // Rewrite the body chunk by chunk. A chunk that cannot be rewritten keeps
    // its original text, so a partial failure still yields a usable scene.
    const chunks = splitSceneBody(scene.content, chunkChars);
    const rewrittenChunks: string[] = [];
    let chunkFailures = 0;
    // Consecutive failures. A provider that is down would otherwise make the
    // scene pay for every one of its chunks before it is given up on.
    let consecutiveFailures = 0;
    let sceneAbandoned = false;
    // Net characters saved by the chunks that were rewritten. Comparing the
    // reassembled lengths instead would count the join separators, and a body
    // whose chunks barely shrank would look unchanged even though it did.
    let savedChars = 0;
    for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex += 1) {
      const chunk = chunks[chunkIndex];
      // A chunk that is already tiny costs a request and a wait but cannot
      // shrink much; it is kept as it is.
      if (chunk.length < minChunkChars) {
        rewrittenChunks.push(chunk);
        continue;
      }
      if (Date.now() >= deadline) {
        budgetExhausted = true;
        failures.push({ filename: scene.filename, reason: "budget_exhausted" });
        // The chunk we did not get to — and every chunk after it — keeps its
        // original text: the budget is gone, so there is nothing to gain from
        // trying again.
        for (const rest of chunks.slice(chunkIndex)) {
          rewrittenChunks.push(rest);
        }
        break;
      }
      let rewritten: string | null | undefined;
      try {
        rewritten = await rewrite({
          filename: scene.filename,
          content: chunk,
          maxTokens,
          timeoutMs,
        });
      } catch (error) {
        chunkFailures += 1;
        consecutiveFailures += 1;
        failures.push({
          filename: scene.filename,
          reason: `chunk_error:${error instanceof Error ? error.message : String(error)}`,
        });
        rewrittenChunks.push(chunk);
        if (consecutiveFailures >= maxChunkFailures) {
          // The provider is not answering. Stop paying for this scene's
          // remaining chunks and keep their original text.
          for (const rest of chunks.slice(chunkIndex + 1)) {
            rewrittenChunks.push(rest);
          }
          failures.push({
            filename: scene.filename,
            reason: `abandoned_after_${consecutiveFailures}_consecutive_failures`,
          });
          break;
        }
        continue;
      }
      const body = normalizeRewrittenBody(typeof rewritten === "string" ? rewritten : "");
      const verdict = validateRewrittenBody(body, chunk, minRatio);
      if (!verdict.ok) {
        chunkFailures += 1;
        consecutiveFailures += 1;
        failures.push({ filename: scene.filename, reason: `chunk_${verdict.reason}` });
        rewrittenChunks.push(chunk);
        if (consecutiveFailures >= maxChunkFailures) {
          for (const rest of chunks.slice(chunkIndex + 1)) {
            rewrittenChunks.push(rest);
          }
          failures.push({
            filename: scene.filename,
            reason: `abandoned_after_${consecutiveFailures}_consecutive_failures`,
          });
          break;
        }
        continue;
      }
      consecutiveFailures = 0;
      savedChars += chunk.length - body.length;
      rewrittenChunks.push(body);
    }

    const rewrittenBody = rewrittenChunks.join("\n\n");
    // A scene where nothing was rewritten, or where the rewrite did not make
    // the body smaller, is left as it was.
    if (chunkFailures >= chunks.length || savedChars <= 0) {
      failures.push({ filename: scene.filename, reason: "no_effective_compression" });
      skipped.push(scene.filename);
      continue;
    }
    scene.content = rewrittenBody;
    compressed.push(scene.filename);
  }

  if (budgetExhausted) {
    failures.push({ filename: "<pass>", reason: "max_duration_exceeded" });
  }

  const state = JSON.stringify(working);
  const finalChars = state.length;
  // A payload that still does not fit is reported with `result: null` so the
  // caller can fall back to the positional trim policy. The failure detail is
  // still reported, so a partially successful pass is diagnosable.
  if (finalChars > limit) {
    return { result: null, failures, compressedScenes: compressed, skippedScenes: skipped };
  }
  return {
    result: {
      state,
      originalChars: original,
      finalChars,
      savedChars: original - finalChars,
      compressedScenes: compressed,
      skippedScenes: skipped,
    },
    failures,
    compressedScenes: compressed,
    skippedScenes: skipped,
  };
}
