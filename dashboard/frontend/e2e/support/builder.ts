import type { Page } from '@playwright/test'

const emptyWasmModule = Buffer.from('0061736d01000000', 'hex')

const mockWasmExecSource = `
(() => {
  const KNOWN_SIGNAL_TYPES = new Set([
    'keyword',
    'embedding',
    'domain',
    'fact_check',
    'user_feedback',
    'preference',
    'language',
    'context',
    'complexity',
    'modality',
    'authz',
    'jailbreak',
    'pii',
  ]);

  function sanitizeDsl(dsl) {
    return dsl
      .split(/\\r?\\n/)
      .filter(line => !line.trim().startsWith('#'))
      .join('\\n');
  }

  function findMatchingBrace(source, openBraceIndex) {
    let depth = 0;
    for (let index = openBraceIndex; index < source.length; index += 1) {
      if (source[index] === '{') depth += 1;
      if (source[index] === '}') {
        depth -= 1;
        if (depth === 0) {
          return index;
        }
      }
    }
    return -1;
  }

  function collectSignals(dsl) {
    const source = sanitizeDsl(dsl);
    const signals = [];
    const pattern = /^SIGNAL\\s+(\\S+)\\s+(\\S+)\\s*\\{/gm;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      signals.push({ type: match[1], name: match[2] });
    }
    return signals;
  }

  function collectRoutes(dsl) {
    return collectRouteBlocks(dsl).map(route => route.name);
  }

  function collectPlugins(dsl) {
    const source = sanitizeDsl(dsl);
    const plugins = [];
    const pattern = /^PLUGIN\\s+(\\S+)\\s+(\\S+)\\s*\\{/gm;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      plugins.push({ name: match[1], pluginType: match[2] });
    }
    return plugins;
  }

  function collectBackends(dsl) {
    const source = sanitizeDsl(dsl);
    const backends = [];
    const pattern = /^BACKEND\\s+(\\S+)\\s+(\\S+)\\s*\\{/gm;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      backends.push({ type: match[1], name: match[2] });
    }
    return backends;
  }

  function collectModels(dsl) {
    const source = sanitizeDsl(dsl);
    const models = [];
    const pattern = /MODEL\\s+([\\s\\S]*?)(?=\\n\\s*(?:ALGORITHM|PLUGIN|ROUTE|BACKEND|GLOBAL)\\b|\\n\\}|$)/gm;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      const refs = [...match[1].matchAll(/"([^"]+)"/g)].map(item => item[1]);
      refs.forEach(model => {
        if (!models.includes(model)) {
          models.push(model);
        }
      });
    }
    return models;
  }

  function parseSimpleWhenExpr(value) {
    const match = value.match(/^(\\w+)\\("([^"]+)"\\)$/);
    if (!match) {
      return null;
    }
    return {
      type: 'signal_ref',
      signalType: match[1],
      signalName: match[2],
      pos: { Line: 1, Column: 1 },
    };
  }

  function collectRouteBlocks(dsl) {
    const source = sanitizeDsl(dsl);
    const routes = [];
    const pattern = /^ROUTE\\s+(\\S+)(?:\\s+\\(description\\s*=\\s*"([^"]*)"\\))?\\s*\\{/gm;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      const openBraceIndex = source.indexOf('{', match.index);
      const closeBraceIndex = findMatchingBrace(source, openBraceIndex);
      const block = closeBraceIndex >= 0
        ? source.slice(match.index, closeBraceIndex + 1)
        : source.slice(match.index);
      const priorityMatch = block.match(/^\\s*PRIORITY\\s+(\\d+)/m);
      const whenMatch = block.match(/^\\s*WHEN\\s+(.+)$/m);
      const modelSectionMatch = block.match(/MODEL\\s+([\\s\\S]*?)(?=\\n\\s*(?:ALGORITHM|PLUGIN|\\})\\b|\\n\\s*\\}|$)/m);
      const pluginMatches = [...block.matchAll(/^\\s*PLUGIN\\s+(\\S+)(?:\\s*\\{([\\s\\S]*?)\\})?\\s*$/gm)];

      routes.push({
        name: match[1],
        description: match[2] || undefined,
        priority: priorityMatch ? Number(priorityMatch[1]) : 100,
        when: whenMatch ? parseSimpleWhenExpr(whenMatch[1].trim()) : null,
        models: modelSectionMatch
          ? [...modelSectionMatch[1].matchAll(/"([^"]+)"(?:\\s*\\(([^)]*)\\))?/g)].map(modelMatch => ({
              model: modelMatch[1],
              pos: { Line: 1, Column: 1 },
            }))
          : [],
        plugins: pluginMatches.map(pluginMatch => ({
          name: pluginMatch[1],
          fields: {},
          pos: { Line: 1, Column: 1 },
        })),
        pos: { Line: 1, Column: 1 },
      });
    }
    return routes;
  }

  function collectDiagnostics(dsl) {
    const source = sanitizeDsl(dsl);
    const diagnostics = [];
    const signals = collectSignals(source);
    const signalNames = new Set(signals.map(signal => signal.name));

    for (const signal of signals) {
      if (!KNOWN_SIGNAL_TYPES.has(signal.type)) {
        diagnostics.push({
          level: 'error',
          message: 'unknown signal type "' + signal.type + '"',
          line: 1,
          column: 1,
        });
      }
    }

    if (source.includes('prompt_guard')) {
      const thresholdMatch = source.match(/threshold:\\s*(\\d+(?:\\.\\d+)?)/);
      if (thresholdMatch) {
        const threshold = Number(thresholdMatch[1]);
        if (Number.isFinite(threshold) && (threshold < 0 || threshold > 1)) {
          diagnostics.push({
            level: 'constraint',
            message: 'prompt_guard threshold must be between 0 and 1',
            line: 1,
            column: 1,
          });
        }
      }
    }

    const whenPattern = /WHEN\\s+(\\w+)\\("([^"]+)"\\)/gm;
    let whenMatch;
    while ((whenMatch = whenPattern.exec(source)) !== null) {
      if (!signalNames.has(whenMatch[2])) {
        diagnostics.push({
          level: 'warning',
          message: 'signal reference "' + whenMatch[2] + '" is not defined',
          line: 1,
          column: 1,
        });
      }
    }

    return diagnostics;
  }

  function buildSymbols(dsl) {
    const source = sanitizeDsl(dsl);
    const signals = collectSignals(source).map(signal => ({ name: signal.name, type: signal.type }));
    const routes = collectRoutes(source);
    return {
      signals,
      models: collectModels(source),
      plugins: collectPlugins(source).map(plugin => plugin.name),
      backends: collectBackends(source).map(backend => ({ name: backend.name, type: backend.type })),
      routes,
    };
  }

  function buildAst(dsl) {
    const source = sanitizeDsl(dsl);
    return {
      signals: collectSignals(source).map(signal => ({
        signalType: signal.type,
        name: signal.name,
        fields: {},
        pos: { Line: 1, Column: 1 },
      })),
      routes: collectRouteBlocks(source),
      plugins: collectPlugins(source).map(plugin => ({
        name: plugin.name,
        pluginType: plugin.pluginType,
        fields: {},
        pos: { Line: 1, Column: 1 },
      })),
      backends: collectBackends(source).map(backend => ({
        backendType: backend.type,
        name: backend.name,
        fields: {},
        pos: { Line: 1, Column: 1 },
      })),
    };
  }

  function compileDsl(dsl) {
    const diagnostics = collectDiagnostics(dsl);
    return JSON.stringify({
      yaml: dsl.trim() ? 'generated: true\\ndsl_length: ' + dsl.length + '\\n' : '',
      crd: '',
      diagnostics,
      ast: buildAst(dsl),
    });
  }

  function validateDsl(dsl) {
    const diagnostics = collectDiagnostics(dsl);
    return JSON.stringify({
      diagnostics,
      errorCount: diagnostics.filter(diagnostic => diagnostic.level === 'error').length,
      symbols: buildSymbols(dsl),
    });
  }

  function parseDsl(dsl) {
    const diagnostics = collectDiagnostics(dsl);
    return JSON.stringify({
      ast: buildAst(dsl),
      diagnostics,
      symbols: buildSymbols(dsl),
      errorCount: diagnostics.filter(diagnostic => diagnostic.level === 'error').length,
    });
  }

  class MockGo {
    constructor() {
      this.importObject = {};
    }

    async run() {
      window.signalCompile = compileDsl;
      window.signalValidate = validateDsl;
      window.signalParseAST = parseDsl;
      window.signalDecompile = (source) => {
        const text = typeof source === 'string' ? source : '';
        const injected = window.__mockBuilderDecompileDsl || '';
        const passthroughDsl = /^\\s*(?:SIGNAL|PLUGIN|ROUTE|BACKEND|GLOBAL)\\b/m.test(text)
          ? text
          : '';
        return JSON.stringify({ dsl: injected || passthroughDsl });
      };
      window.signalFormat = (dsl) => JSON.stringify({ dsl });
    }
  }

  window.Go = MockGo;
})();
`

export async function mockBuilderWasm(page: Page): Promise<void> {
  await page.route('**/wasm_exec.js', async route => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/javascript' },
      body: mockWasmExecSource,
    })
  })

  await page.route('**/signal-compiler.wasm', async route => {
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'application/wasm',
        ETag: 'mock-builder-wasm',
      },
      body: emptyWasmModule,
    })
  })
}
