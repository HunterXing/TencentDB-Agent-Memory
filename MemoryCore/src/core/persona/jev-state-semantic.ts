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
 *   JEV_SEMANTIC_MAX_TOKENS     max tokens for the rewrite call, default 16000
 *   JEV_SEMANTIC_TIMEOUT_MS      timeout for the rewrite call, default 120000
 *   JEV_SEMANTIC_MIN_RATIO       minimum fraction of the original that must
 *                                survive, default 0.2 — guards against a
 *                                model that returns an empty or stub answer
 *   JEV_SEMANTIC_MAX_SCENES      how many scenes to compress per pass,
 *                                default 4 — larger sets are handled by
 *                                repeated passes
 */

import type { JevStatePayload } from "./jev-state-trim.js";

export const JEV_SEMANTIC_MAX_TOKENS_DEFAULT = 16_000;
export const JEV_SEMANTIC_TIMEOUT_MS_DEFAULT = 120_000;
export const JEV_SEMANTIC_MIN_RATIO_DEFAULT = 0.2;
export const JEV_SEMANTIC_MAX_SCENES_DEFAULT = 4;

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
  const tail = rewritten.slice(-24).trim();
  if (tail.length > 0 && !/[.!?。！？：:;；)\]}】」』"'\`]$/.test(tail)) {
    return { ok: false, reason: "possibly_truncated" };
  }
  return { ok: true };
}

/**
 * Build the rewrite prompt. Kept separate so it can be asserted in tests.
 */
export function buildSemanticRewritePrompt(filename: string, content: string): string {
  return [
    "你是一个记忆库整理器。下面是一份「某个用户的某个主题」的长篇记忆文档。",
    "",
    "任务：把它改写成一份**更短但信息不丢**的版本。",
    "",
    "必须遵守：",
    "1. **删除过时/已被推翻的内容**：已被后续状态取代的旧事实、已完成的临时任务记录、重复的复述流水、只记热度的元记录。",
    "2. **保留当前仍然成立的**：硬事实、规则/铁律、当前架构与状态、偏好、未决的问题。",
    "3. **保留章节结构**：用 markdown 二级标题分区（章节与原文对应，不要凭空新增或删除章节）。",
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
    env?: NodeJS.ProcessEnv;
  } = {},
): Promise<JevSemanticOutcome> {
  const env = opts.env ?? process.env;
  const maxTokens = opts.maxTokens ?? resolveJevSemanticMaxTokens(env);
  const timeoutMs = opts.timeoutMs ?? resolveJevSemanticTimeoutMs(env);
  const minRatio = opts.minRatio ?? resolveJevSemanticMinRatio(env);
  const maxScenes = opts.maxScenes ?? resolveJevSemanticMaxScenes(env);

  const original = measureJevState(payload);
  const scenes = payload.changed_scenes.map((scene) => ({ ...scene }));
  const working: JevStatePayload = { ...payload, changed_scenes: scenes };
  const compressed: string[] = [];
  const skipped: string[] = [];
  const failures: Array<{ filename: string; reason: string }> = [];

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

  for (const index of order) {
    if (measureJevState(working) <= limit) break;
    if (compressed.length >= maxScenes) break;
    const scene = scenes[index];
    if (!scene.content || scene.content.length === 0) continue;

    let rewritten: string | null | undefined;
    try {
      rewritten = await rewrite({
        filename: scene.filename,
        content: scene.content,
        maxTokens,
        timeoutMs,
      });
    } catch (error) {
      failures.push({
        filename: scene.filename,
        reason: error instanceof Error ? error.message : String(error),
      });
      skipped.push(scene.filename);
      continue;
    }

    const body = normalizeRewrittenBody(typeof rewritten === "string" ? rewritten : "");
    const verdict = validateRewrittenBody(body, scene.content, minRatio);
    if (!verdict.ok) {
      failures.push({ filename: scene.filename, reason: verdict.reason });
      skipped.push(scene.filename);
      continue;
    }
    scene.content = body;
    compressed.push(scene.filename);
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
