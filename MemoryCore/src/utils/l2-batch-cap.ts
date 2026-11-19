/**
 * L2 批量上限（2026-11-19 引入）。
 *
 * 背景：L2 场景抽取本身是增量查询（游标 updatedAfter），但冷启动与失败重试后
 * 游标会落后，原实现把查询到的全部 L1 记忆一次性塞进 prompt（实测 639 条 /
 * 26.8 万字符 / 单次 13-18 万 token），一天可烧掉 8 亿 prompt token。
 *
 * 这里按「条数 + 字符」双上限截断：保持原有 ASC 顺序（老记录优先处理），
 * 调用方只把本批最后一条的 updatedAt 作为新游标，剩余记录留待下一轮。
 */

export interface L2BatchRecord {
  content: string;
}

export interface L2BatchCapOptions {
  /** 单批最多处理的记录条数 */
  maxRecords: number;
  /** 单批累计 content 字符上限（与条数上限同时生效，先到者为准） */
  maxChars: number;
}

export interface L2BatchCapResult<T extends L2BatchRecord> {
  records: T[];
  /** 是否发生了截断（有记录被留到下一轮） */
  capped: boolean;
  /** 本批累计字符数 */
  chars: number;
  /** 被留待下一轮的记录条数 */
  remaining: number;
}

/** 按条数 + 字符双上限截断 L2 输入批次。至少保留 1 条（单条超限也照发）。 */
export function capL2Batch<T extends L2BatchRecord>(
  records: T[],
  opts: L2BatchCapOptions,
): L2BatchCapResult<T> {
  const maxRecords = Math.max(1, Math.floor(opts.maxRecords) || 1);
  const maxChars = Math.max(1, Math.floor(opts.maxChars) || 1);

  const totalChars = records.reduce((n, r) => n + (r.content?.length ?? 0), 0);
  if (records.length <= maxRecords && totalChars <= maxChars) {
    return { records, capped: false, chars: totalChars, remaining: 0 };
  }

  const kept: T[] = [];
  let chars = 0;
  for (const rec of records) {
    if (kept.length >= maxRecords) break;
    const len = rec.content?.length ?? 0;
    // 至少保留一条，避免单条超长导致本批为空、游标无法推进
    if (kept.length > 0 && chars + len > maxChars) break;
    chars += len;
    kept.push(rec);
  }

  return {
    records: kept,
    capped: kept.length < records.length,
    chars,
    remaining: records.length - kept.length,
  };
}
