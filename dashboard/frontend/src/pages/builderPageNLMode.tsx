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
import {
  useResizableHeight,
  useSplitPaneWidth,
} from "./builderPageResizeHooks";
import { buildIntentReviewCards } from "./builderPageNLReviewSupport";
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
type ApplyState = "idle" | "applying" | "applied";

const MIN_WORKSPACE_PANE_WIDTH = 340;
const WORKSPACE_SPLITTER_WIDTH = 14;

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
  const [applyState, setApplyState] = useState<ApplyState>("idle");
  const reviewEngaged = Boolean(plannerResult);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const reviewSectionRef = useRef<HTMLElement | null>(null);
  const appliedBannerRef = useRef<HTMLDivElement | null>(null);
  const preferredPromptPaneRatio = reviewEngaged ? 0.46 : 0.54;
  const {
    width: promptPaneWidth,
    isDragging: isResizingWorkspace,
    handleDragStart: handleWorkspaceResizeStart,
    handleKeyDown: handleWorkspaceResizeKeyDown,
  } = useSplitPaneWidth({
    containerRef: workspaceRef,
    preferredRatio: preferredPromptPaneRatio,
    minStartWidth: MIN_WORKSPACE_PANE_WIDTH,
    minEndWidth: MIN_WORKSPACE_PANE_WIDTH,
    dividerWidth: WORKSPACE_SPLITTER_WIDTH,
  });
  const {
    height: reviewViewportHeight,
    isDragging: isResizingReviewViewport,
    handleDragStart: handleReviewViewportResizeStart,
    handleKeyDown: handleReviewViewportResizeKeyDown,
  } = useResizableHeight({
    initialHeight: 430,
    minHeight: 260,
    getMaxHeight: () =>
      Math.max(
        420,
        typeof window === "undefined" ? 720 : window.innerHeight - 160,
      ),
  });

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

  const contextStatItems = useMemo(
    () => [
      { label: "Signals", value: currentContextSummary.signals },
      { label: "Routes", value: currentContextSummary.routes },
      { label: "Plugins", value: currentContextSummary.plugins },
      { label: "Backends", value: currentContextSummary.backends },
    ],
    [currentContextSummary],
  );

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
  const reviewCards = useMemo(
    () => buildIntentReviewCards(plannerResult, ast),
    [ast, plannerResult],
  );

  const draftLineCount = useMemo(() => countLines(draftDsl), [draftDsl]);
  const currentLineCount = useMemo(() => countLines(dslSource), [dslSource]);
  const draftMatchesBuilder = useMemo(
    () => normalizeDslForCompare(draftDsl) === normalizeDslForCompare(dslSource),
    [draftDsl, dslSource],
  );

  useEffect(() => {
    if (!draftDsl.trim()) {
      setApplyState("idle");
      return;
    }
    if (applyState === "applying" && draftMatchesBuilder) {
      setApplyState("applied");
      setActiveReviewTab("changes");
      return;
    }
    if (!draftMatchesBuilder && applyState === "applied") {
      setApplyState("idle");
    }
  }, [applyState, draftDsl, draftMatchesBuilder]);

  useEffect(() => {
    if (!plannerResult) {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      reviewSectionRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [plannerResult]);

  useEffect(() => {
    if (applyState !== "applied") {
      return;
    }
    const frame = window.requestAnimationFrame(() => {
      appliedBannerRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [applyState]);

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
    setApplyState("idle");

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
    const nextDraftDiff = buildDraftDiff(dslSource, preparedDraft.draft.dsl);
    setDraftDsl(preparedDraft.draft.dsl);
    setDraftSummary(preparedDraft.draft.summary);
    setActiveReviewTab(selectDefaultReviewTab(nextDraftDiff));
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
    setApplyState("idle");
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
    wasmReady &&
    !!draftDsl.trim() &&
    draftIssueCount === 0 &&
    !readonlyMode &&
    applyState !== "applied";
  const canRepairDraft =
    wasmReady &&
    !!draftDsl.trim() &&
    draftIssueCount > 0 &&
    capabilities?.plannerAvailable;
  const plannerBadgeTitle = capabilities?.plannerAvailable
    ? "Preview feature"
    : "Planner unavailable";
  const plannerBadgeMeta = capabilities?.plannerAvailable
    ? formatPlannerBackendLabel(capabilities?.plannerBackend)
    : "No planner backend configured";
  const previewBadgeTitle = capabilities?.preview
    ? "Review before apply"
    : "Stable workflow";
  const previewBadgeMeta = capabilities?.preview
    ? "Draft-only workflow"
    : "Planner can write into the working Builder state";
  const workspaceStyle = {
    "--prompt-pane-width": promptPaneWidth
      ? `${promptPaneWidth}px`
      : undefined,
  } as React.CSSProperties;
  const workspaceMaxPromptWidth = Math.max(
    MIN_WORKSPACE_PANE_WIDTH,
    (workspaceRef.current?.clientWidth ?? promptPaneWidth ?? 854) -
      WORKSPACE_SPLITTER_WIDTH -
      MIN_WORKSPACE_PANE_WIDTH,
  );
  const reviewViewportMaxHeight = Math.max(
    420,
    typeof window === "undefined" ? 720 : window.innerHeight - 160,
  );

  const handleApplyDraft = useCallback(() => {
    if (!canApplyDraft) {
      return;
    }
    setApplyState("applying");
    onApplyDraft(draftDsl);
  }, [canApplyDraft, draftDsl, onApplyDraft]);

  return (
    <div className={styles.container} data-testid="builder-nl-mode">
      <div className={styles.header}>
        <div className={styles.headerMain}>
          <div>
            <div className={styles.title}>Natural Language Drafting</div>
            <div className={styles.subtitle}>
              Builder stays on canonical DSL. The planner only proposes a
              reviewed draft before anything touches the working config.
            </div>
          </div>
          <div className={styles.badgeRow}>
            <span
              className={`${styles.badge} ${styles.badgePrimary}`}
              title={capabilities?.plannerBackend ?? "planner unavailable"}
              data-testid="builder-nl-badge-planner"
            >
              <span className={styles.badgeDot} aria-hidden="true" />
              <span className={styles.badgeCopy}>
                <strong>{plannerBadgeTitle}</strong>
                <small>{plannerBadgeMeta}</small>
              </span>
            </span>
            <span
              className={`${styles.badge} ${styles.badgeInfo}`}
              data-testid="builder-nl-badge-mode"
            >
              <span className={styles.badgeDot} aria-hidden="true" />
              <span className={styles.badgeCopy}>
                <strong>{previewBadgeTitle}</strong>
                <small>{previewBadgeMeta}</small>
              </span>
            </span>
            {readonlyMode && <span className={styles.badgeWarn}>readonly</span>}
          </div>
        </div>
        <div className={styles.heroInfoRow}>
          {contextStatItems.map((item) => (
            <div key={item.label} className={styles.heroStatCard}>
              <span className={styles.statLabel}>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
          <div className={styles.heroNoteCard}>
            <span className={styles.statLabel}>Workflow</span>
            <strong>Prompt to Review to Apply</strong>
            <p>
              Start with one change, inspect assumptions and diff, then apply
              only when the canonical draft looks right.
            </p>
          </div>
        </div>
      </div>

      <div
        ref={workspaceRef}
        className={`${styles.grid} ${styles.gridSplit}`}
        style={workspaceStyle}
      >
        <section
          className={`${styles.card} ${styles.promptCard}`}
          data-testid="builder-nl-prompt-panel"
        >
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.sectionTitle}>Prompt</h3>
              <p className={styles.sectionMeta}>
                Draft against the current Builder DSL context.
              </p>
            </div>
            <button className={styles.secondaryButton} onClick={handleReset}>
              New session
            </button>
          </div>

          <div className={styles.composerCard}>
            <div className={styles.composerHeader}>
              <div>
                <div className={styles.contextLabel}>Compose one change</div>
                <div className={styles.sectionMeta}>
                  Ask for a single create, update, delete, or fix against the
                  current Builder DSL.
                </div>
              </div>
              <div className={styles.composerHints}>
                <span className={styles.composerHint}>Single change</span>
                <span className={styles.composerHint}>Exact names work best</span>
              </div>
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
          </div>

          {surfaceError && <div className={styles.errorBox}>{surfaceError}</div>}

          <div className={styles.promptRailHeader}>
            <div>
              <div className={styles.contextLabel}>Prompt support</div>
              <div className={styles.sectionMeta}>
                Use quick starts and supported construct hints to keep the draft
                inside the preview planner’s working set.
              </div>
            </div>
          </div>

          <div className={styles.promptLowerGrid}>
            <div className={styles.quickStartCard}>
              <div className={styles.exampleHeader}>
                <div className={styles.contextLabel}>Quick starts</div>
                <span className={styles.hint}>One change per turn</span>
              </div>
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

            <div className={styles.supportStack}>
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
                <div className={styles.tipList}>
                  <div className={styles.tipItem}>Describe one change per turn.</div>
                  <div className={styles.tipItem}>
                    Mention exact route, model, signal, or plugin names when editing existing config.
                  </div>
                  <div className={styles.tipItem}>
                    Drafts stay review-only until you apply them to Builder.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div
          className={`${styles.layoutResizeHandle} ${
            isResizingWorkspace ? styles.layoutResizeHandleActive : ""
          }`}
          data-testid="builder-nl-layout-splitter"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize NL prompt and review panels"
          aria-valuemin={MIN_WORKSPACE_PANE_WIDTH}
          aria-valuemax={workspaceMaxPromptWidth}
          aria-valuenow={Math.round(promptPaneWidth ?? 0)}
          tabIndex={0}
          onMouseDown={handleWorkspaceResizeStart}
          onKeyDown={handleWorkspaceResizeKeyDown}
        >
          <div className={styles.layoutResizeHandleLine} />
        </div>

        <section
          className={`${styles.card} ${!plannerResult ? styles.reviewCardEmpty : ""}`}
          data-testid="builder-nl-review"
          ref={reviewSectionRef}
        >
          <div className={styles.sectionHeader}>
            <div>
              <h3 className={styles.sectionTitle}>Review</h3>
              <p className={styles.sectionMeta}>
                Current Builder DSL stays unchanged until you apply a reviewed draft.
              </p>
            </div>
          </div>

          {!plannerResult && (
            <div className={styles.emptyReviewBody}>
              <div className={styles.emptyState}>
                <strong>Draft review will appear here.</strong>
                <span>
                  Generate one change from the prompt panel, then inspect the
                  planner interpretation, assumptions, semantic changes, and
                  canonical DSL result before applying it.
                </span>
              </div>
              <div className={styles.emptySteps}>
                <div className={styles.emptyStep}>
                  <span className={styles.emptyStepIndex}>1</span>
                  <div>
                    <strong>Describe one change</strong>
                    <p>Use exact route, model, signal, or plugin names when editing existing config.</p>
                  </div>
                </div>
                <div className={styles.emptyStep}>
                  <span className={styles.emptyStepIndex}>2</span>
                  <div>
                    <strong>Review assumptions and changes</strong>
                    <p>Check the semantic change summary first, then inspect the diff or full draft if needed.</p>
                  </div>
                </div>
                <div className={styles.emptyStep}>
                  <span className={styles.emptyStepIndex}>3</span>
                  <div>
                    <strong>Apply only when ready</strong>
                    <p>Applying replaces the working Builder DSL, then recompiles it so deploy preview uses the new draft.</p>
                  </div>
                </div>
              </div>
              <div className={styles.emptyPreviewFrame}>
                <div className={styles.emptyPreviewHeader}>
                  <div>
                    <div className={styles.contextLabel}>Review canvas</div>
                    <div className={styles.sectionMeta}>
                      Changes stay review-only until you apply them to Builder.
                    </div>
                  </div>
                  <span className={styles.hint}>Current DSL remains untouched</span>
                </div>
                <div className={styles.emptyPreviewTabs}>
                  <span className={styles.emptyPreviewTabActive}>Changes</span>
                  <span className={styles.emptyPreviewTab}>Diff</span>
                  <span className={styles.emptyPreviewTab}>Full draft</span>
                </div>
                <div className={styles.emptyPreviewGrid}>
                  <div className={styles.emptyPreviewCard}>
                    <span className={styles.statLabel}>Planner interpretation</span>
                    <p>See the planner’s understanding of your request and any default assumptions it made.</p>
                  </div>
                  <div className={styles.emptyPreviewCard}>
                    <span className={styles.statLabel}>Semantic changes</span>
                    <p>Check the route, signal, plugin, or backend changes before you trust the full DSL diff.</p>
                  </div>
                  <div className={styles.emptyPreviewCard}>
                    <span className={styles.statLabel}>Canonical DSL result</span>
                    <p>Inspect the diff or full draft that will replace the working Builder DSL after apply.</p>
                  </div>
                </div>
                <div className={styles.emptyPreviewSkeleton}>
                  <div className={`${styles.emptyPreviewLine} ${styles.emptyPreviewLineStrong}`} />
                  <div className={styles.emptyPreviewLine} />
                  <div className={`${styles.emptyPreviewLine} ${styles.emptyPreviewLineShort}`} />
                  <div className={styles.emptyPreviewSeparator} />
                  <div className={`${styles.emptyPreviewLine} ${styles.emptyPreviewLineStrong}`} />
                  <div className={`${styles.emptyPreviewLine} ${styles.emptyPreviewLineShort}`} />
                </div>
              </div>
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

              {applyState === "applied" && (
                <div
                  ref={appliedBannerRef}
                  className={styles.appliedBanner}
                  data-testid="builder-nl-applied-banner"
                >
                  Draft applied to Builder. The current working DSL now matches this reviewed draft.
                </div>
              )}

              {draftDiagnostics.length > 0 && (
                <div className={styles.diagnosticList}>
                  {draftDiagnostics.slice(0, 5).map((diag, index) => (
                    <div key={`${diag.message}-${index}`} className={styles.diagnosticItem}>
                      <strong>{diag.level}</strong>: {diag.message}
                    </div>
                  ))}
                </div>
              )}

              <div
                className={styles.reviewFrame}
                data-testid="builder-nl-review-frame"
                style={{ height: `${reviewViewportHeight}px` }}
              >
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

                <div
                  className={styles.reviewViewport}
                  data-testid="builder-nl-review-viewport"
                >
                  {activeReviewTab === "changes" && (
                    <div className={styles.reviewPanel}>
                      <div className={styles.semanticHighlights}>
                        <div className={styles.highlightCard}>
                          <span className={styles.statLabel}>Review mode</span>
                          <strong>
                            {draftDiff?.removals
                              ? "Update existing Builder DSL"
                              : "Append reviewed draft to Builder DSL"}
                          </strong>
                          <p>
                            {draftMatchesBuilder
                              ? "Builder already matches this draft."
                              : "Current Builder DSL remains unchanged until you apply."}
                          </p>
                        </div>
                        <div className={styles.highlightCard}>
                          <span className={styles.statLabel}>Primary change</span>
                          <strong>{draftSummary[0] ?? "Draft change"}</strong>
                          <p>
                            {draftDiff?.additions ?? 0} added line(s), {draftDiff?.removals ?? 0} removed line(s).
                          </p>
                        </div>
                        <div className={styles.highlightCard}>
                          <span className={styles.statLabel}>Apply effect</span>
                          <strong>
                            {applyState === "applied" ? "Applied to Builder" : "Replaces working DSL"}
                          </strong>
                          <p>
                            {applyState === "applied"
                              ? "Compile and deploy now use the applied draft."
                              : "Apply writes this reviewed draft into Builder, then recompiles it."}
                          </p>
                        </div>
                      </div>

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
                        <div className={styles.contextLabel}>Structured review</div>
                        <div className={styles.intentReviewGrid}>
                          {reviewCards.map((card) => (
                            <div key={card.id} className={styles.intentReviewCard}>
                              <div className={styles.intentReviewHeader}>
                                <span className={styles.intentReviewKicker}>
                                  {card.kicker}
                                </span>
                                <strong className={styles.intentReviewTitle}>
                                  {card.title}
                                </strong>
                              </div>
                              <p className={styles.intentReviewSummary}>
                                {card.summary}
                              </p>
                              <div className={styles.intentReviewDetails}>
                                {card.details.map((detail) => (
                                  <div
                                    key={`${card.id}-${detail}`}
                                    className={styles.intentReviewDetail}
                                  >
                                    {detail}
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div>
                        <div className={styles.contextLabel}>Change summary</div>
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
                      <div className={styles.diffHeader}>
                        <span>Op</span>
                        <span>Current</span>
                        <span>Draft</span>
                        <span>Line</span>
                      </div>
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
              </div>

              <div
                className={`${styles.reviewResizeHandle} ${
                  isResizingReviewViewport ? styles.reviewResizeHandleActive : ""
                }`}
                data-testid="builder-nl-review-resize"
                role="separator"
                aria-orientation="horizontal"
                aria-label="Resize NL draft review viewport"
                aria-valuemin={260}
                aria-valuemax={reviewViewportMaxHeight}
                aria-valuenow={Math.round(reviewViewportHeight)}
                tabIndex={0}
                onMouseDown={handleReviewViewportResizeStart}
                onKeyDown={handleReviewViewportResizeKeyDown}
              >
                <div className={styles.reviewResizeHandleLine} />
              </div>

              <div className={styles.reviewActionBar}>
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
                  onClick={handleApplyDraft}
                  disabled={!canApplyDraft}
                >
                  {readonlyMode
                    ? "Readonly"
                    : applyState === "applying"
                      ? "Applying..."
                      : applyState === "applied"
                        ? "Applied to Builder"
                        : "Apply to Builder"}
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

function formatPlannerBackendLabel(backend: string | null | undefined): string {
  if (!backend) {
    return "No backend configured";
  }
  if (backend === "preview-rulebased") {
    return "Rule-based planner";
  }
  return backend.replace(/[-_]/g, " ");
}

function selectDefaultReviewTab(diff: ReturnType<typeof buildDraftDiff>): ReviewTab {
  if (!diff.hasChanges || (diff.additions > 0 && diff.removals === 0)) {
    return "changes";
  }
  return "diff";
}

function countLines(value: string): number {
  if (!value.trim()) {
    return 0;
  }
  return value.replace(/\r\n/g, "\n").replace(/\n$/, "").split("\n").length;
}

function normalizeDslForCompare(value: string): string {
  return value.replace(/\r\n/g, "\n").trim();
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
