export type DraftDiffKind = "context" | "add" | "remove" | "separator";

export interface DraftDiffLine {
  kind: DraftDiffKind;
  leftNumber: number | null;
  rightNumber: number | null;
  text: string;
}

export interface DraftDiff {
  lines: DraftDiffLine[];
  additions: number;
  removals: number;
  hasChanges: boolean;
}

interface RawDiffLine {
  kind: Exclude<DraftDiffKind, "separator">;
  leftNumber: number | null;
  rightNumber: number | null;
  text: string;
}

const DIFF_CONTEXT_RADIUS = 2;

export function buildDraftDiff(before: string, after: string): DraftDiff {
  const beforeLines = normalizeLines(before);
  const afterLines = normalizeLines(after);
  const rawLines = buildRawDiff(beforeLines, afterLines);
  const additions = rawLines.filter((line) => line.kind === "add").length;
  const removals = rawLines.filter((line) => line.kind === "remove").length;

  return {
    lines: compressContext(rawLines, DIFF_CONTEXT_RADIUS),
    additions,
    removals,
    hasChanges: additions > 0 || removals > 0,
  };
}

function normalizeLines(value: string): string[] {
  const normalized = value.replace(/\r\n/g, "\n");
  if (!normalized) {
    return [];
  }
  const lines = normalized.split("\n");
  if (lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines;
}

function buildRawDiff(beforeLines: string[], afterLines: string[]): RawDiffLine[] {
  const lcs = buildLcsTable(beforeLines, afterLines);
  const output: RawDiffLine[] = [];

  let beforeIndex = 0;
  let afterIndex = 0;
  let leftNumber = 1;
  let rightNumber = 1;

  while (beforeIndex < beforeLines.length && afterIndex < afterLines.length) {
    if (beforeLines[beforeIndex] === afterLines[afterIndex]) {
      output.push({
        kind: "context",
        leftNumber,
        rightNumber,
        text: beforeLines[beforeIndex],
      });
      beforeIndex += 1;
      afterIndex += 1;
      leftNumber += 1;
      rightNumber += 1;
      continue;
    }

    if (lcs[beforeIndex + 1][afterIndex] >= lcs[beforeIndex][afterIndex + 1]) {
      output.push({
        kind: "remove",
        leftNumber,
        rightNumber: null,
        text: beforeLines[beforeIndex],
      });
      beforeIndex += 1;
      leftNumber += 1;
      continue;
    }

    output.push({
      kind: "add",
      leftNumber: null,
      rightNumber,
      text: afterLines[afterIndex],
    });
    afterIndex += 1;
    rightNumber += 1;
  }

  while (beforeIndex < beforeLines.length) {
    output.push({
      kind: "remove",
      leftNumber,
      rightNumber: null,
      text: beforeLines[beforeIndex],
    });
    beforeIndex += 1;
    leftNumber += 1;
  }

  while (afterIndex < afterLines.length) {
    output.push({
      kind: "add",
      leftNumber: null,
      rightNumber,
      text: afterLines[afterIndex],
    });
    afterIndex += 1;
    rightNumber += 1;
  }

  return output;
}

function buildLcsTable(beforeLines: string[], afterLines: string[]): number[][] {
  const rows = beforeLines.length + 1;
  const cols = afterLines.length + 1;
  const table = Array.from({ length: rows }, () => Array(cols).fill(0));

  for (let i = beforeLines.length - 1; i >= 0; i -= 1) {
    for (let j = afterLines.length - 1; j >= 0; j -= 1) {
      if (beforeLines[i] === afterLines[j]) {
        table[i][j] = table[i + 1][j + 1] + 1;
      } else {
        table[i][j] = Math.max(table[i + 1][j], table[i][j + 1]);
      }
    }
  }

  return table;
}

function compressContext(
  lines: RawDiffLine[],
  contextRadius: number,
): DraftDiffLine[] {
  const changedIndexes = lines
    .map((line, index) => (line.kind === "context" ? -1 : index))
    .filter((index) => index >= 0);

  if (changedIndexes.length === 0) {
    return lines.map((line) => ({ ...line }));
  }

  const keepIndexes = new Set<number>();
  for (const index of changedIndexes) {
    const start = Math.max(0, index - contextRadius);
    const end = Math.min(lines.length - 1, index + contextRadius);
    for (let cursor = start; cursor <= end; cursor += 1) {
      keepIndexes.add(cursor);
    }
  }

  const output: DraftDiffLine[] = [];
  let index = 0;
  while (index < lines.length) {
    if (keepIndexes.has(index)) {
      output.push({ ...lines[index] });
      index += 1;
      continue;
    }

    let hiddenCount = 0;
    while (index < lines.length && !keepIndexes.has(index)) {
      hiddenCount += 1;
      index += 1;
    }

    output.push({
      kind: "separator",
      leftNumber: null,
      rightNumber: null,
      text: `${hiddenCount} unchanged line(s) hidden`,
    });
  }

  return output;
}
