/**
 * Vitest setup: keep the ambient environment out of the tests.
 *
 * The production core container runs with JEV_STATE_OVERFLOW_POLICY=semantic
 * (and the other JEV_* knobs). A test that asserts the *unset* behavior
 * would silently read that inherited value and change what it is testing:
 * the same suite passed on a developer machine and failed inside the
 * container. The individual tests that care now pin the value they need,
 * and this setup makes sure nothing else inherits one by accident.
 */
import { beforeEach } from "vitest";

const JEV_ENV_KEYS = [
  "JEV_STATE_LIMIT_CHARS",
  "JEV_STATE_OVERFLOW_POLICY",
  "JEV_SEMANTIC_MAX_TOKENS",
  "JEV_SEMANTIC_TIMEOUT_MS",
  "JEV_SEMANTIC_MIN_RATIO",
  "JEV_SEMANTIC_MAX_SCENES",
  "JEV_SEMANTIC_CHUNK_CHARS",
  "JEV_SEMANTIC_MIN_CHUNK_CHARS",
  "JEV_SEMANTIC_MAX_DURATION_MS",
  "JEV_SEMANTIC_MAX_CHUNK_FAILURES",
  "JEV_SEMANTIC_TARGET_RATIO",
];

beforeEach(() => {
  for (const key of JEV_ENV_KEYS) {
    delete process.env[key];
  }
});
