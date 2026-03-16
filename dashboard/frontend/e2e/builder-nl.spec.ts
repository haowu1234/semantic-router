import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { mockBuilderWasm } from './support/builder'

const schemaVersion = '2026-03-13'
const sessionId = 'builder-nl-session-1'
const turnId = 'builder-nl-turn-1'
const expiresAt = '2026-03-13T10:00:00Z'

const baseRouterYaml = `
signals: {}
decisions: []
providers:
  models: []
plugins: {}
`

const existingRouteDsl = `
SIGNAL keyword urgent_signal {
  operator: "any"
  keywords: ["urgent"]
}

PLUGIN blocker fast_response {
  message: "blocked"
}

ROUTE support_route {
  PRIORITY 75

  WHEN keyword("urgent_signal")

  MODEL "gpt-4o-mini"

  PLUGIN blocker
}
`

const nlSchema = {
  version: schemaVersion,
  signals: [
    {
      typeName: 'keyword',
      description: 'Match explicit keywords or regex-like literals.',
      fields: [
        { key: 'keywords', label: 'Keywords', type: 'string[]', required: true },
        { key: 'operator', label: 'Operator', type: 'select', options: ['any', 'all'] },
      ],
    },
  ],
  plugins: [],
  algorithms: [],
  backends: [],
}

function buildCapabilities(readonlyMode: boolean) {
  return {
    enabled: true,
    preview: true,
    plannerAvailable: true,
    plannerBackend: 'preview-rulebased',
    schemaVersion,
    supportedOperations: ['generate', 'modify'],
    supportedConstructs: ['signal', 'route', 'plugin', 'backend'],
    supportedSignalTypes: ['keyword', 'embedding'],
    supportedPluginTypes: ['semantic_cache', 'memory', 'fast_response', 'system_prompt'],
    supportedBackendTypes: ['vllm_endpoint'],
    supportedAlgorithmTypes: [],
    supportsClarification: true,
    supportsSessionApi: true,
    supportsStreaming: false,
    supportsApply: !readonlyMode,
    readonlyMode,
  }
}

function buildReadySignalTurn() {
  return {
    sessionId,
    turnId,
    schemaVersion,
    expiresAt,
    result: {
      status: 'ready',
      explanation: 'Create a keyword signal urgent_signal for 2 keyword(s).',
      warnings: [
        {
          code: 'default_keyword_operator',
          message: 'Using operator "any" because the prompt did not request "all".',
        },
      ],
      intentIr: {
        version: '1.0',
        operation: 'generate',
        intents: [
          {
            type: 'signal',
            signal_type: 'keyword',
            name: 'urgent_signal',
            fields: {
              operator: 'any',
              keywords: ['urgent', 'asap'],
            },
          },
        ],
      },
    },
  }
}

function buildInvalidThresholdTurn() {
  return {
    sessionId,
    turnId,
    schemaVersion,
    expiresAt,
    result: {
      status: 'ready',
      explanation: 'Add prompt-guard settings to the global block.',
      intentIr: {
        version: '1.0',
        operation: 'generate',
        intents: [
          {
            type: 'global',
            fields: {
              prompt_guard: {
                enabled: true,
                threshold: 1.5,
              },
            },
          },
        ],
      },
    },
  }
}

function buildRouteModelUpdateTurn() {
  return {
    sessionId,
    turnId: 'builder-nl-turn-route-update',
    schemaVersion,
    expiresAt,
    result: {
      status: 'ready',
      explanation: 'Update route support_route to use model gpt-4.1-mini.',
      intentIr: {
        version: '1.0',
        operation: 'modify',
        intents: [
          {
            type: 'modify',
            action: 'update',
            target_construct: 'route',
            target_name: 'support_route',
            changes: {
              models: [{ model: 'gpt-4.1-mini' }],
            },
          },
        ],
      },
    },
  }
}

async function mockBuilderNLPreview(page: Page, readonlyMode: boolean) {
  await mockBuilderWasm(page)
  await mockAuthenticatedAppShell(page, {
    settings: {
      readonlyMode,
      setupMode: false,
      platform: '',
      envoyUrl: '',
    },
  })

  await page.route('**/api/router/config/yaml', async route => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'text/plain' },
      body: baseRouterYaml,
    })
  })

  await page.route('**/api/builder/nl/capabilities', async route => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildCapabilities(readonlyMode)),
    })
  })

  await page.route('**/api/builder/nl/schema', async route => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(nlSchema),
    })
  })
}

test.describe('Builder NL mode', () => {
  test('generates a signal draft and can open it in DSL mode', async ({ page }) => {
    let sessionPayload: Record<string, unknown> | null = null
    let turnPayload: Record<string, unknown> | null = null

    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      sessionPayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      turnPayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildReadySignalTurn()),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    const promptInput = page.getByTestId('builder-nl-prompt')
    const generateButton = page.getByRole('button', { name: 'Generate draft' })

    await expect(page.getByTestId('builder-nl-mode')).toBeVisible()
    await expect(page.getByTestId('builder-nl-badge-planner')).toContainText('Preview feature')
    await expect(page.getByTestId('builder-nl-badge-mode')).toContainText('Review before apply')
    await expect(page.getByTestId('builder-nl-review')).toContainText('Describe one change')
    await expect(generateButton).toBeDisabled()

    await promptInput.fill('Create a keyword signal named urgent_signal with keywords "urgent", "asap"')
    await expect(generateButton).toBeEnabled()
    await generateButton.click()

    await expect.poll(() => Boolean(sessionPayload)).toBe(true)
    await expect.poll(() => Boolean(turnPayload)).toBe(true)
    expect((sessionPayload?.context as Record<string, unknown> | undefined)?.baseDsl).toBeDefined()
    expect((turnPayload?.context as Record<string, unknown> | undefined)?.baseDsl).toBeDefined()
    expect(turnPayload?.prompt).toBe(
      'Create a keyword signal named urgent_signal with keywords "urgent", "asap"',
    )

    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Create a keyword signal urgent_signal for 2 keyword(s).',
    )
    await expect(page.getByTestId('builder-nl-review')).toContainText('Draft validates cleanly')
    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Append reviewed draft to Builder DSL',
    )
    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Keywords: urgent, asap',
    )
    await page.getByTestId('builder-nl-review-tab-diff').click()
    await expect(page.getByTestId('builder-nl-diff-preview')).toContainText(
      'SIGNAL keyword urgent_signal',
    )
    await page.getByTestId('builder-nl-review-tab-draft').click()
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'SIGNAL keyword urgent_signal',
    )

    await page.getByTestId('builder-nl-open-in-dsl').click()

    await expect(page.getByTestId('builder-nl-mode')).toHaveCount(0)
    await page.getByTestId('builder-output-tab-dsl').click()
    await expect(page.getByTestId('builder-output-code')).toContainText(
      'SIGNAL keyword urgent_signal',
    )
    await expect(page.getByTestId('builder-output-code')).toContainText(
      'keywords: ["urgent", "asap"]',
    )
  })

  test('readonly mode still reviews drafts but does not allow applying them', async ({ page }) => {
    await mockBuilderNLPreview(page, true)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(true),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildReadySignalTurn()),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    await page.getByTestId('builder-nl-prompt').fill(
      'Create a keyword signal named urgent_signal with keywords "urgent", "asap"',
    )
    await page.getByRole('button', { name: 'Generate draft' }).click()

    await page.getByTestId('builder-nl-review-tab-draft').click()
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'SIGNAL keyword urgent_signal',
    )
    await expect(page.getByTestId('builder-nl-apply-draft')).toBeDisabled()
    await expect(page.getByTestId('builder-nl-apply-draft')).toContainText('Readonly')
    await expect(page.getByTestId('builder-nl-open-in-dsl')).toBeEnabled()
  })

  test('workspace panes can be resized and review viewport exposes a resize control', async ({ page }) => {
    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildReadySignalTurn()),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    const promptPanel = page.getByTestId('builder-nl-prompt-panel')
    const reviewPanel = page.getByTestId('builder-nl-review')

    const beforePromptBox = await promptPanel.boundingBox()
    const beforeReviewBox = await reviewPanel.boundingBox()
    expect(beforePromptBox).not.toBeNull()
    expect(beforeReviewBox).not.toBeNull()

    await page.getByTestId('builder-nl-layout-splitter').evaluate(node => {
      node.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }))
      node.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }))
    })

    const afterPromptBox = await promptPanel.boundingBox()
    const afterReviewBox = await reviewPanel.boundingBox()
    expect(afterPromptBox).not.toBeNull()
    expect(afterReviewBox).not.toBeNull()
    expect(afterPromptBox?.width ?? 0).toBeLessThan((beforePromptBox?.width ?? 0) - 40)
    expect(afterReviewBox?.width ?? 0).toBeGreaterThan((beforeReviewBox?.width ?? 0) + 40)

    await page.getByTestId('builder-nl-prompt').fill(
      'Create a keyword signal named urgent_signal with keywords "urgent", "asap"',
    )
    await page.getByRole('button', { name: 'Generate draft' }).click()

    const viewport = page.getByTestId('builder-nl-review-viewport')
    await viewport.scrollIntoViewIfNeeded()
    await expect(page.getByTestId('builder-nl-review-resize')).toBeVisible()
    await expect(page.getByTestId('builder-nl-review-frame')).toBeVisible()
    await expect(page.getByTestId('builder-nl-review-resize')).toHaveAttribute(
      'aria-orientation',
      'horizontal',
    )
  })

  test('schema-invalid planner drafts are rejected in review', async ({ page }) => {
    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          turnId: 'builder-nl-turn-invalid',
          schemaVersion,
          expiresAt,
          result: {
            status: 'ready',
            explanation: 'Create a signal the planner thinks exists.',
            intentIr: {
              version: '1.0',
              operation: 'generate',
              intents: [
                {
                  type: 'signal',
                  signal_type: 'ghost_signal',
                  name: 'bad_signal',
                  fields: {
                    keywords: ['urgent'],
                  },
                },
              ],
            },
          },
        }),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    await page.getByTestId('builder-nl-prompt').fill('Create a ghost signal')
    await page.getByRole('button', { name: 'Generate draft' }).click()

    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Planner returned an invalid structured draft',
    )
    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'unsupported signal type ghost_signal',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).toHaveCount(0)
    await expect(page.getByTestId('builder-nl-apply-draft')).toHaveCount(0)
  })

  test('invalid drafts can be retried with validator diagnostics', async ({ page }) => {
    let turnCallCount = 0
    let repairPayload: Record<string, unknown> | null = null
    let repairFollowupPayload: Record<string, unknown> | null = null

    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      turnCallCount += 1
      if (turnCallCount === 1) {
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildInvalidThresholdTurn()),
        })
        return
      }

      if (turnCallCount === 2) {
        repairPayload = route.request().postDataJSON() as Record<string, unknown>
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sessionId,
            turnId: 'builder-nl-turn-2',
            schemaVersion,
            expiresAt,
            result: {
              status: 'needs_clarification',
              explanation: 'The validator flagged an invalid threshold value.',
              clarification: {
                question: 'What threshold should replace the invalid value?',
                options: [
                  { id: '0.7', label: '0.7', description: 'Balanced threshold.' },
                  { id: '0.8', label: '0.8', description: 'Stricter threshold.' },
                ],
              },
            },
          }),
        })
        return
      }

      repairFollowupPayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          turnId: 'builder-nl-turn-3',
          schemaVersion,
          expiresAt,
          result: {
            status: 'ready',
            explanation: 'Repair prompt-guard threshold by setting it to 0.7.',
            intentIr: {
              version: '1.0',
              operation: 'fix',
              intents: [
                {
                  type: 'global',
                  fields: {
                    prompt_guard: {
                      enabled: true,
                      threshold: 0.7,
                    },
                  },
                },
              ],
            },
          },
        }),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    await page.getByTestId('builder-nl-prompt').fill('Add prompt guard threshold 1.5')
    await page.getByRole('button', { name: 'Generate draft' }).click()

    await expect(page.getByTestId('builder-nl-review')).toContainText('Needs review:')
    await expect(page.getByTestId('builder-nl-repair-draft')).toBeEnabled()
    await expect(page.getByTestId('builder-nl-apply-draft')).toBeDisabled()

    await page.getByTestId('builder-nl-repair-draft').click()

    await expect.poll(() => Boolean(repairPayload)).toBe(true)
    expect(repairPayload?.modeHint).toBe('fix')
    expect((repairPayload?.context as Record<string, unknown> | undefined)?.baseDsl).toEqual(
      expect.stringContaining('threshold: 1.5'),
    )
    expect(
      Array.isArray((repairPayload?.context as Record<string, unknown> | undefined)?.diagnostics),
    ).toBe(true)

    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'What threshold should replace the invalid value?',
    )

    await page.getByRole('button', { name: '0.7' }).click()

    await expect.poll(() => Boolean(repairFollowupPayload)).toBe(true)
    expect(repairFollowupPayload?.modeHint).toBe('fix')
    expect((repairFollowupPayload?.context as Record<string, unknown> | undefined)?.baseDsl).toEqual(
      expect.stringContaining('threshold: 1.5'),
    )

    await page.getByTestId('builder-nl-review-tab-draft').click()
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText('threshold: 0.7')
    await expect(page.getByTestId('builder-nl-apply-draft')).toBeEnabled()
  })

  test('route updates preserve existing route condition and plugin refs', async ({ page }) => {
    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildRouteModelUpdateTurn()),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'Import' }).click()
    await page.getByPlaceholder('Paste YAML config here...').fill(existingRouteDsl)
    await page.getByRole('button', { name: /^Import$/ }).last().click()
    await page.getByTestId('builder-output-tab-dsl').click()
    await expect(page.getByTestId('builder-output-code')).toContainText('ROUTE support_route')
    await expect(page.getByTestId('builder-output-code')).toContainText('PLUGIN blocker')
    await page.getByRole('button', { name: 'NL' }).click()

    await page.getByTestId('builder-nl-prompt').fill(
      'Update route support_route to use model gpt-4.1-mini',
    )
    await page.getByRole('button', { name: 'Generate draft' }).click()

    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Update route support_route to use model gpt-4.1-mini.',
    )
    await page.getByTestId('builder-nl-review-tab-changes').click()
    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Model refs: gpt-4o-mini -> gpt-4.1-mini',
    )
    await expect(page.getByTestId('builder-nl-review')).toContainText(
      'Unspecified route settings stay unchanged.',
    )
    await page.getByTestId('builder-nl-review-tab-diff').click()
    await expect(page.getByTestId('builder-nl-diff-preview')).toContainText(
      'gpt-4.1-mini',
    )
    await page.getByTestId('builder-nl-review-tab-draft').click()
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'ROUTE support_route',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'PRIORITY 75',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'WHEN keyword("urgent_signal")',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'MODEL "gpt-4.1-mini"',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).toContainText(
      'PLUGIN blocker',
    )
    await expect(page.getByTestId('builder-nl-draft-preview')).not.toContainText(
      'MODEL "gpt-4o-mini"',
    )
  })

  test('applied NL drafts can open the existing deploy preview flow', async ({ page }) => {
    let previewPayload: Record<string, unknown> | null = null

    await mockBuilderNLPreview(page, false)

    await page.route('**/api/builder/nl/sessions', async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          schemaVersion,
          expiresAt,
          capabilities: buildCapabilities(false),
        }),
      })
    })

    await page.route(`**/api/builder/nl/sessions/${sessionId}/turns`, async route => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildReadySignalTurn()),
      })
    })

    await page.route('**/api/router/config/deploy/preview', async route => {
      previewPayload = route.request().postDataJSON() as Record<string, unknown>
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current: 'signals: {}',
          preview: 'signals:\n  keywords:\n    - name: urgent_signal',
        }),
      })
    })

    await page.goto('/builder')
    await page.getByRole('button', { name: 'NL' }).click()

    await page.getByTestId('builder-nl-prompt').fill(
      'Create a keyword signal named urgent_signal with keywords "urgent", "asap"',
    )
    await page.getByRole('button', { name: 'Generate draft' }).click()
    await page.getByTestId('builder-nl-apply-draft').click()

    await expect(page.getByTestId('builder-nl-applied-banner')).toContainText(
      'Draft applied to Builder.',
    )

    await expect(page.getByRole('button', { name: 'Deploy' })).toBeEnabled()
    await page.getByRole('button', { name: 'Deploy' }).click()

    await expect.poll(() => Boolean(previewPayload)).toBe(true)
    expect(typeof previewPayload?.yaml).toBe('string')
    expect((previewPayload?.yaml as string).length).toBeGreaterThan(0)

    await expect(page.getByText('Deploy to Router')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy Now' })).toBeEnabled()
    await expect(page.getByText('A backup of the current config will be created before deployment.')).toBeVisible()
  })
})
