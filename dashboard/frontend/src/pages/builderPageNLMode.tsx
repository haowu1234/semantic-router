import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { wasmBridge } from "@/lib/wasm";
import type { ASTProgram, Diagnostic, SymbolTable } from "@/types/dsl";
import type {
  NLAuthoringCapabilities,
  NLClarificationOption,
  NLOperationMode,
  NLPlannerResult,
  NLSessionContext,
  NLSchemaManifest,
} from "@/types/nl";
import {
  createNLAuthoringSession,
  createNLAuthoringTurn,
  getNLAuthoringCapabilities,
  getNLAuthoringSchema,
} from "@/utils/nlAuthoringApi";

import { buildDraftDiff } from "./builderPageNLDiff";
import { preparePlannerDraft } from "./builderPageNLDraft";
import styles from "./builderPageNLMode.module.css";

interface BuilderNLModeProps {
  dslSource: string;
  ast: ASTProgram | null;
  symbols: SymbolTable | null;
  diagnostics: Diagnostic[];
  wasmReady: boolean;
  readonlyMode: boolean;
  onApplyDraft: (draftDsl: string) => void;
  onOpenDraftInDSL: (draftDsl: string) => void;
}

interface RunTurnOptions {
  modeHint?: NLOperationMode;
  context?: NLSessionContext;
}

type ReviewTab = "changes" | "diff" | "draft";

const BuilderNLMode: React.FC<BuilderNLModeProps> = ({
  dslSource,
  ast,
  symbols,
  diagnostics,
  wasmReady,
  readonlyMode,
  onApplyDraft,
  onOpenDraftInDSL,
}) => {
  const [prompt, setPrompt] = useState("");
  const [submittedPrompt, setSubmittedPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [surfaceError, setSurfaceError] = useState<string | null>(null);
  const [capabilities, setCapabilities] =
    useState<NLAuthoringCapabilities | null>(null);
  const [schema, setSchema] = useState<NLSchemaManifest | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionExpiresAt, setSessionExpiresAt] = useState<string | null>(null);
  const lastTurnOptionsRef = useRef<RunTurnOptions>({});
  const [plannerResult, setPlannerResult] = useState<NLPlannerResult | null>(
    null,
  );
  const [draftDsl, setDraftDsl] = useState<string>("");
  const [draftDiagnostics, setDraftDiagnostics] = useState<Diagnostic[]>([]);
  const [draftSummary, setDraftSummary] = useState<string[]>([]);
  const [activeReviewTab, setActiveReviewTab] = useState<ReviewTab>("changes");

  useEffect(() => {
    let cancelled = false;

    const loadSurface = async () => {
      try {
        const [nextCapabilities, nextSchema] = await Promise.all([
          getNLAuthoringCapabilities(),
          getNLAuthoringSchema(),
        ]);
        if (cancelled) {
          return;
        }
        setCapabilities(nextCapabilities);
        setSchema(nextSchema);
      } catch (err) {
        if (!cancelled) {
          setSurfaceError(normalizeError(err));
        }
      }
    };

    void loadSurface();
    return () => {
      cancelled = true;
    };
  }, []);

  const currentContextSummary = useMemo(() => {
    const errorCount = diagnostics.filter((diag) => diag.level === "error").length;
    return {
      signals: symbols?.signals.length ?? 0,
      routes: symbols?.routes.length ?? 0,
      plugins: symbols?.plugins.length ?? 0,
      backends: symbols?.backends.length ?? 0,
      errors: errorCount,
    };
  }, [diagnostics, symbols]);

  const examplePrompts = useMemo(() => {
    const examples = [
      `Create a keyword signal named urgent_signal with keywords "urgent", "asap"`,
      `Create a fast_response plugin named safe_block with "I cannot help with that request."`,
    ];
    if (symbols?.models?.length) {
      examples.unshift(
        `Create route support_route for model ${symbols.models[0]}`,
      );
    }
    if (symbols?.signals?.length) {
      examples.push(`Delete signal ${symbols.signals[0].name}`);
    }
    return examples.slice(0, 4);
  }, [symbols]);

  const supportedSignalEntries = useMemo(() => {
    if (!schema) {
      return [];
    }
    const supportedTypes = new Set(capabilities?.supportedSignalTypes ?? []);
    const entries =
      supportedTypes.size > 0
        ? schema.signals.filter((entry) => supportedTypes.has(entry.typeName))
        : schema.signals;
    return entries.slice(0, 5);
  }, [capabilities?.supportedSignalTypes, schema]);

  const draftDiff = useMemo(() => {
    if (!draftDsl.trim()) {
      return null;
    }
    return buildDraftDiff(dslSource, draftDsl);
  }, [draftDsl, dslSource]);

  const draftLineCount = useMemo(() => countLines(draftDsl), [draftDsl]);
  const currentLineCount = useMemo(() => countLines(dslSource), [dslSource]);

  const validateDraft = useCallback((nextDraft: string) => {
    if (!wasmReady || !nextDraft.trim()) {
      setDraftDiagnostics([]);
      return;
    }

    const result = wasmBridge.validate(nextDraft);
    setDraftDiagnostics(result.diagnostics ?? []);
  }, [wasmReady]);

  const buildBuilderContext = useCallback((): NLSessionContext => ({
    baseDsl: dslSource,
    symbols: symbols ?? undefined,
    diagnostics,
  }), [diagnostics, dslSource, symbols]);

  const ensureSession = useCallback(async () => {
    if (sessionId) {
      return sessionId;
    }

    const response = await createNLAuthoringSession({
      schemaVersion: capabilities?.schemaVersion,
      context: buildBuilderContext(),
    });
    setSessionId(response.sessionId);
    setSessionExpiresAt(response.expiresAt ?? null);
    setCapabilities(response.capabilities);
    return response.sessionId;
  }, [buildBuilderContext, capabilities?.schemaVersion, sessionId]);

  const consumePlannerResult = useCallback((result: NLPlannerResult) => {
    setPlannerResult(result);
    setDraftDsl("");
    setDraftDiagnostics([]);
    setDraftSummary([]);
    setActiveReviewTab("changes");

    const preparedDraft = preparePlannerDraft(dslSource, result, schema, ast);
    if (preparedDraft.error) {
      setPlannerResult({
        status: "error",
        explanation: result.explanation,
        warnings: [
          ...(result.warnings ?? []),
          {
            code: "invalid_intent_ir",
            message: "Planner output could not be prepared into a canonical Builder draft.",
          },
        ],
        error: preparedDraft.error,
      });
      return;
    }

    if (!preparedDraft.draft) {
      return;
    }
    setDraftDsl(preparedDraft.draft.dsl);
    setDraftSummary(preparedDraft.draft.summary);
    setActiveReviewTab("diff");
    validateDraft(preparedDraft.draft.dsl);
  }, [ast, dslSource, schema, validateDraft]);

  const runTurn = useCallback(
    async (
      nextPrompt: string,
      options: RunTurnOptions = {},
      allowRetry = true,
    ) => {
      const resolvedPrompt = nextPrompt.trim();
      if (!resolvedPrompt) {
        return;
      }

      setLoading(true);
      setSurfaceError(null);
      setSubmittedPrompt(resolvedPrompt);
      lastTurnOptionsRef.current = options;

      try {
        const currentSessionId = await ensureSession();
        const response = await createNLAuthoringTurn(currentSessionId, {
          prompt: resolvedPrompt,
          modeHint: options.modeHint,
          schemaVersion: capabilities?.schemaVersion,
          context: options.context ?? buildBuilderContext(),
        });

        setSessionExpiresAt(response.expiresAt ?? null);
        consumePlannerResult(response.result);
      } catch (err) {
        const message = normalizeError(err);
        if (
          allowRetry &&
          (message.includes("session not found") || message.includes('"error":"not_found"'))
        ) {
          setSessionId(null);
          await runTurn(resolvedPrompt, options, false);
          return;
        }
        setSurfaceError(message);
      } finally {
        setLoading(false);
      }
    },
    [
      capabilities?.schemaVersion,
      buildBuilderContext,
      consumePlannerResult,
      ensureSession,
    ],
  );

  const handleSubmit = useCallback(async () => {
    await runTurn(prompt);
  }, [prompt, runTurn]);

  const handleRepairDraft = useCallback(async () => {
    if (!draftDsl.trim() || draftDiagnostics.length === 0) {
      return;
    }

    await runTurn(submittedPrompt || prompt, {
      modeHint: "fix",
      context: {
        baseDsl: draftDsl,
        symbols: symbols ?? undefined,
        diagnostics: draftDiagnostics,
      },
    });
  }, [draftDiagnostics, draftDsl, prompt, runTurn, submittedPrompt, symbols]);

  const handleClarificationOption = useCallback(
    async (option: NLClarificationOption) => {
      const nextPrompt = composeClarificationPrompt(
        submittedPrompt || prompt,
        plannerResult,
        option,
      );
      setPrompt(nextPrompt);
      await runTurn(nextPrompt, lastTurnOptionsRef.current);
    },
    [plannerResult, prompt, runTurn, submittedPrompt],
  );

  const handleReset = useCallback(() => {
    setPrompt("");
    setSubmittedPrompt("");
    setPlannerResult(null);
    setDraftDsl("");
    setDraftDiagnostics([]);
    setDraftSummary([]);
    setSessionId(null);
    setSessionExpiresAt(null);
    setSurfaceError(null);
    lastTurnOptionsRef.current = {};
  }, []);

  const draftErrorCount = draftDiagnostics.filter(
    (diag) => diag.level === "error",
  ).length;
  const draftConstraintCount = draftDiagnostics.filter(
    (diag) => diag.level === "constraint",
  ).length;
  const draftWarningCount = draftDiagnostics.filter(
    (diag) => diag.level === "warning",
  ).length;
  const draftIssueCount =
    draftErrorCount + draftConstraintCount + draftWarningCount;
  const canApplyDraft =
    wasmReady && !!draftDsl.trim() && draftIssueCount === 0 && !readonlyMode;
  const canRepairDraft =
    wasmReady &&
    !!draftDsl.trim() &&
    draftIssueCount > 0 &&
    capabilities?.plannerAvailable;

  return (
    <div className={styles.container} data-testid="builder-nl-mode">
      <div className={styles.header}>
        <div>
          <div className={styles.title}>Natural Language Drafting</div>
          <div className={styles.subtitle}>
            Builder stays on canonical DSL. The planner only proposes a draft.
          </div>
        </div>
        <div className={styles.badgeRow}>
          <span
            className={styles.badge}
            title={capabilities?.plannerBackend ?? "planner unavailable"}
          >
            {capabilities?.plannerAvailable ? "Preview planner" : "Planner unavailable"}
          </span>
          <span className={styles.badge}>
            {capabilities?.preview ? "Draft-only preview" : "stable"}
          </span>
          {readonlyMode && <span className={styles.badgeWarn}>readonly</span>}
        </div>
      </div>

      <div className={styles.grid}>
        <section className={styles.card}>
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.sectionTitle}>Prompt</h3>
              <p className={styles.sectionMeta}>
                Context: {currentContextSummary.signals} signals,{" "}
                {currentContextSummary.routes} routes,{" "}
                {currentContextSummary.plugins} plugins,{" "}
                {currentContextSummary.backends} backends
              </p>
            </div>
            <button className={styles.secondaryButton} onClick={handleReset}>
              New session
            </button>
          </div>

          <textarea
            className={styles.textarea}
            data-testid="builder-nl-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder='Describe one change, for example: Create route support_route for model gpt-4o-mini'
            rows={6}
          />

          <div className={styles.actionRow}>
            <button
              className={styles.primaryButton}
              onClick={() => void handleSubmit()}
              disabled={
                loading || !wasmReady || !prompt.trim() || !capabilities?.enabled
              }
            >
              {loading ? "Planning..." : "Generate draft"}
            </button>
            <span className={styles.hint}>
              {sessionExpiresAt
                ? `Session expires ${new Date(sessionExpiresAt).toLocaleTimeString()}`
                : "Session starts on first turn"}
            </span>
          </div>

          {surfaceError && <div className={styles.errorBox}>{surfaceError}</div>}

          <div className={styles.exampleGroup}>
            <div className={styles.sectionMeta}>Examples</div>
            <div className={styles.exampleList}>
              {examplePrompts.map((example) => (
                <button
                  key={example}
                  className={styles.exampleChip}
                  onClick={() => setPrompt(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.contextGrid}>
            <div className={styles.contextCard}>
              <div className={styles.contextLabel}>Supported constructs</div>
              <div className={styles.pillRow}>
                {(capabilities?.supportedConstructs ?? []).map((construct) => (
                  <span key={construct} className={styles.pill}>
                    {construct}
                  </span>
                ))}
              </div>
            </div>
            <div className={styles.contextCard}>
              <div className={styles.contextLabel}>Signal types</div>
              <div className={styles.pillRow}>
                {supportedSignalEntries.map((entry) => (
                  <span key={entry.typeName} className={styles.pill}>
                    {entry.typeName}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className={styles.guidanceCard}>
            <div className={styles.contextLabel}>Best results</div>
            <ul className={styles.guidanceList}>
              <li>Describe one change per turn.</li>
              <li>Mention exact route, model, signal, or plugin names when editing existing config.</li>
              <li>Drafts stay review-only until you apply them to Builder.</li>
            </ul>
          </div>
        </section>

        <section className={styles.card} data-testid="builder-nl-review">
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.sectionTitle}>Review</h3>
              <p className={styles.sectionMeta}>
                Current Builder DSL stays unchanged until you apply a reviewed draft.
              </p>
            </div>
          </div>

          {!plannerResult && (
            <div className={styles.emptyState}>
              <strong>Draft review will appear here.</strong>
              <span>Generate one change from the prompt panel, then review assumptions, proposed changes, and the canonical DSL result before applying it.</span>
            </div>
          )}

          {plannerResult?.explanation && (
            <div>
              <div className={styles.contextLabel}>Planner interpretation</div>
              <div className={styles.explanation}>{plannerResult.explanation}</div>
            </div>
          )}

          {plannerResult?.warnings?.length ? (
            <div>
              <div className={styles.contextLabel}>Assumptions made</div>
              <div className={styles.warningList}>
                {plannerResult.warnings.map((warning) => (
                  <div key={`${warning.code}-${warning.message}`} className={styles.warningItem}>
                    <strong>{warning.code}</strong>: {warning.message}
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {plannerResult?.status === "needs_clarification" &&
            plannerResult.clarification && (
              <div className={styles.clarificationCard}>
                <div className={styles.contextLabel}>
                  {plannerResult.clarification.question}
                </div>
                <div className={styles.optionList}>
                  {plannerResult.clarification.options.map((option) => (
                    <button
                      key={option.id}
                      className={styles.optionButton}
                      onClick={() => void handleClarificationOption(option)}
                    >
                      <span>{option.label}</span>
                      {option.description && (
                        <small>{option.description}</small>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

          {plannerResult?.status === "unsupported" && (
            <div className={styles.errorBox}>
              {plannerResult.error || "The preview planner cannot satisfy this request yet."}
            </div>
          )}

          {plannerResult?.status === "error" && plannerResult.error && (
            <div className={styles.errorBox}>{plannerResult.error}</div>
          )}

          {!!draftDsl && (
            <>
              <div className={styles.validationRow}>
                <span
                  className={
                    draftIssueCount === 0 ? styles.validBadge : styles.invalidBadge
                  }
                >
                  {draftIssueCount === 0
                    ? "Draft validates cleanly"
                    : formatDraftIssueSummary(
                        draftErrorCount,
                        draftConstraintCount,
                        draftWarningCount,
                      )}
                </span>
                  {currentContextSummary.errors > 0 && (
                  <span className={styles.hint}>
                    Current DSL has {currentContextSummary.errors} existing error(s).
                  </span>
                )}
              </div>

              {draftDiagnostics.length > 0 && (
                <div className={styles.diagnosticList}>
                  {draftDiagnostics.slice(0, 5).map((diag, index) => (
                    <div key={`${diag.message}-${index}`} className={styles.diagnosticItem}>
                      <strong>{diag.level}</strong>: {diag.message}
                    </div>
                  ))}
                </div>
              )}

              <div className={styles.reviewFrame}>
                <div className={styles.reviewToolbar}>
                  <div>
                    <div className={styles.contextLabel}>Draft review</div>
                    <div className={styles.sectionMeta}>
                      Compare the current Builder DSL with the draft before you apply it.
                    </div>
                  </div>
                  <div className={styles.tabRow} data-testid="builder-nl-review-tabs">
                    <button
                      className={activeReviewTab === "changes" ? styles.activeTab : styles.tabButton}
                      data-testid="builder-nl-review-tab-changes"
                      onClick={() => setActiveReviewTab("changes")}
                    >
                      Changes
                    </button>
                    <button
                      className={activeReviewTab === "diff" ? styles.activeTab : styles.tabButton}
                      data-testid="builder-nl-review-tab-diff"
                      onClick={() => setActiveReviewTab("diff")}
                    >
                      Diff
                    </button>
                    <button
                      className={activeReviewTab === "draft" ? styles.activeTab : styles.tabButton}
                      data-testid="builder-nl-review-tab-draft"
                      onClick={() => setActiveReviewTab("draft")}
                    >
                      Full draft
                    </button>
                  </div>
                </div>

                {activeReviewTab === "changes" && (
                  <div className={styles.reviewPanel}>
                    <div className={styles.statGrid}>
                      <div className={styles.statCard}>
                        <span className={styles.statLabel}>Current DSL</span>
                        <strong>{currentLineCount} line(s)</strong>
                      </div>
                      <div className={styles.statCard}>
                        <span className={styles.statLabel}>Draft DSL</span>
                        <strong>{draftLineCount} line(s)</strong>
                      </div>
                      <div className={styles.statCard}>
                        <span className={styles.statLabel}>Added</span>
                        <strong>+{draftDiff?.additions ?? 0}</strong>
                      </div>
                      <div className={styles.statCard}>
                        <span className={styles.statLabel}>Removed</span>
                        <strong>-{draftDiff?.removals ?? 0}</strong>
                      </div>
                    </div>

                    <div>
                      <div className={styles.contextLabel}>Proposed changes</div>
                      <div className={styles.summaryList}>
                        {draftSummary.map((item) => (
                          <div key={item} className={styles.summaryItem}>
                            {item}
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className={styles.infoBox}>
                      This draft has not changed the current Builder DSL yet. Use <strong>Apply to Builder</strong> to replace the working DSL with this reviewed draft.
                    </div>
                  </div>
                )}

                {activeReviewTab === "diff" && (
                  <div
                    className={styles.diffPreview}
                    data-testid="builder-nl-diff-preview"
                  >
                    {draftDiff?.lines.map((line, index) => (
                      <div
                        key={`${line.kind}-${line.leftNumber ?? "x"}-${line.rightNumber ?? "y"}-${index}`}
                        className={
                          line.kind === "add"
                            ? styles.diffAdd
                            : line.kind === "remove"
                              ? styles.diffRemove
                              : line.kind === "separator"
                                ? styles.diffSeparator
                                : styles.diffContext
                        }
                      >
                        <span className={styles.diffMarker}>
                          {line.kind === "add"
                            ? "+"
                            : line.kind === "remove"
                              ? "-"
                              : line.kind === "separator"
                                ? "..."
                                : " "}
                        </span>
                        <span className={styles.diffNumber}>{line.leftNumber ?? ""}</span>
                        <span className={styles.diffNumber}>{line.rightNumber ?? ""}</span>
                        <code className={styles.diffText}>{line.text || " "}</code>
                      </div>
                    ))}
                  </div>
                )}

                {activeReviewTab === "draft" && (
                  <pre
                    className={styles.codePreview}
                    data-testid="builder-nl-draft-preview"
                  >
                    {draftDsl}
                  </pre>
                )}
              </div>

              <div className={styles.actionRow}>
                {canRepairDraft && (
                  <button
                    className={styles.secondaryButton}
                    data-testid="builder-nl-repair-draft"
                    onClick={() => void handleRepairDraft()}
                    disabled={loading}
                  >
                    Retry with diagnostics
                  </button>
                )}
                <button
                  className={styles.primaryButton}
                  data-testid="builder-nl-apply-draft"
                  onClick={() => onApplyDraft(draftDsl)}
                  disabled={!canApplyDraft}
                >
                  {readonlyMode ? "Readonly" : "Apply to Builder"}
                </button>
                <button
                  className={styles.secondaryButton}
                  data-testid="builder-nl-open-in-dsl"
                  onClick={() => onOpenDraftInDSL(draftDsl)}
                  disabled={!draftDsl.trim()}
                >
                  Edit in DSL
                </button>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
};

function composeClarificationPrompt(
  basePrompt: string,
  plannerResult: NLPlannerResult | null,
  option: NLClarificationOption,
): string {
  const question = plannerResult?.clarification?.question.toLowerCase() ?? "";
  const trimmedBase = basePrompt.trim();

  if (question.includes("which construct")) {
    return `${trimmedBase}. Focus on the ${option.label.toLowerCase()} construct.`;
  }
  if (question.includes("what kind of signal")) {
    return `${trimmedBase}. Use a ${option.id} signal.`;
  }
  if (question.includes("what kind of plugin")) {
    return `${trimmedBase}. Use a ${option.id} plugin.`;
  }
  if (question.includes("what kind of algorithm")) {
    return `${trimmedBase}. Use a ${option.id} algorithm.`;
  }
  if (question.includes("what kind of backend")) {
    return `${trimmedBase}. Use a ${option.id} backend.`;
  }
  if (question.includes("which model")) {
    return `${trimmedBase}. Use model ${option.label}.`;
  }
  if (question.includes("which signal")) {
    return `${trimmedBase}. Use signal ${option.label}.`;
  }
  if (question.includes("threshold")) {
    return `${trimmedBase}. Use threshold ${option.id}.`;
  }
  if (question.includes("keywords")) {
    return `${trimmedBase}. ${option.description ?? option.label}`;
  }
  return `${trimmedBase}. ${option.label}.`;
}

function normalizeError(err: unknown): string {
  if (err instanceof Error && err.message) {
    return parseJsonErrorMessage(err.message) ?? err.message;
  }
  if (typeof err === "string") {
    return parseJsonErrorMessage(err) ?? err;
  }
  return "Unknown NL authoring error";
}

function formatDraftIssueSummary(
  errorCount: number,
  constraintCount: number,
  warningCount: number,
): string {
  const parts: string[] = [];
  if (errorCount > 0) {
    parts.push(`${errorCount} error(s)`);
  }
  if (constraintCount > 0) {
    parts.push(`${constraintCount} constraint(s)`);
  }
  if (warningCount > 0) {
    parts.push(`${warningCount} warning(s)`);
  }
  return `Needs review: ${parts.join(", ")}`;
}

function countLines(value: string): number {
  if (!value.trim()) {
    return 0;
  }
  return value.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n").length;
}

function parseJsonErrorMessage(raw: string): string | null {
  try {
    const parsed = JSON.parse(raw) as { message?: string; error?: string };
    return parsed.message ?? parsed.error ?? null;
  } catch {
    return null;
  }
}

export default BuilderNLMode;
