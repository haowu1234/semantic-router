/**
 * NL Schema Registry
 *
 * A declarative registry that maps every DSL construct (signal, plugin, algorithm, backend)
 * to its natural language metadata. When a new type is added to the DSL, registering it here
 * automatically enables:
 *   1. NL entity extraction (nl_triggers → type recognition)
 *   2. LLM system prompt generation (nl_description + fields → prompt context)
 *   3. Intent IR validation (fields schema → constraint checking)
 *
 * Data sources:
 *   - Field schemas: derived from getSignalFieldSchema / getPluginFieldSchema / getAlgorithmFieldSchema
 *   - NL metadata: curated trigger words and descriptions
 */

import {
  type FieldSchema,
  ALGORITHM_DESCRIPTIONS,
  PLUGIN_DESCRIPTIONS,
  getSignalFieldSchema,
  getPluginFieldSchema,
  getAlgorithmFieldSchema,
} from './dslMutations'

// ─────────────────────────────────────────────
// Schema Entry Interface
// ─────────────────────────────────────────────

export interface NLSchemaEntry {
  /** DSL construct category */
  construct: 'signal' | 'plugin' | 'algorithm' | 'backend'
  /** Type name as used in DSL (e.g., "keyword", "confidence", "semantic_cache") */
  type_name: string
  /** Natural language trigger words/phrases — used for entity extraction */
  nl_triggers: string[]
  /** One-sentence description for LLM context */
  nl_description: string
  /** Example NL phrases that use this type */
  nl_examples: string[]
  /** Typed field schema (mirrors dslMutations FieldSchema) */
  fields: FieldSchema[]
  /** Backend types this entity requires (e.g., RAG plugin needs vector_store) */
  requires_backend?: string[]
  /** Signal types this entity requires */
  requires_signal?: string[]
}

// ─────────────────────────────────────────────
// Registry Class
// ─────────────────────────────────────────────

export class NLSchemaRegistry {
  private entries: Map<string, NLSchemaEntry> = new Map()

  /** Register a schema entry. Key = `${construct}:${type_name}` */
  register(entry: NLSchemaEntry): void {
    const key = `${entry.construct}:${entry.type_name}`
    this.entries.set(key, entry)
  }

  /** Register multiple entries at once */
  registerAll(entries: NLSchemaEntry[]): void {
    for (const entry of entries) {
      this.register(entry)
    }
  }

  /** Get a specific entry */
  get(construct: string, typeName: string): NLSchemaEntry | undefined {
    return this.entries.get(`${construct}:${typeName}`)
  }

  /** Get all entries for a construct type */
  getByConstruct(construct: 'signal' | 'plugin' | 'algorithm' | 'backend'): NLSchemaEntry[] {
    return Array.from(this.entries.values()).filter(e => e.construct === construct)
  }

  /** Get all entries */
  getAll(): NLSchemaEntry[] {
    return Array.from(this.entries.values())
  }

  /** Check if a type name is registered */
  has(construct: string, typeName: string): boolean {
    return this.entries.has(`${construct}:${typeName}`)
  }

  /**
   * Find entries matching any of the given trigger words.
   * Returns entries sorted by number of matching triggers (descending).
   */
  findByTriggers(words: string[]): NLSchemaEntry[] {
    const lowerWords = words.map(w => w.toLowerCase())
    const scored: Array<{ entry: NLSchemaEntry; score: number }> = []

    for (const entry of this.entries.values()) {
      let score = 0
      for (const trigger of entry.nl_triggers) {
        const lowerTrigger = trigger.toLowerCase()
        for (const word of lowerWords) {
          if (word.includes(lowerTrigger) || lowerTrigger.includes(word)) {
            score++
          }
        }
      }
      if (score > 0) {
        scored.push({ entry, score })
      }
    }

    return scored
      .sort((a, b) => b.score - a.score)
      .map(s => s.entry)
  }

  /**
   * Build the "available types" section of the LLM system prompt.
   * Automatically regenerated when registry changes.
   */
  buildSystemPromptSection(): string {
    const sections: string[] = []

    // Signal types
    const signals = this.getByConstruct('signal')
    if (signals.length > 0) {
      sections.push('### Signal Types (for SIGNAL declarations and WHEN conditions)')
      for (const s of signals) {
        const requiredFields = s.fields.filter(f => f.required).map(f => f.key)
        const optionalFields = s.fields.filter(f => !f.required).map(f => f.key)
        let line = `- **${s.type_name}**: ${s.nl_description}`
        if (requiredFields.length > 0) {
          line += `\n  Required fields: ${requiredFields.join(', ')}`
        }
        if (optionalFields.length > 0) {
          line += `\n  Optional fields: ${optionalFields.join(', ')}`
        }
        sections.push(line)
      }
    }

    // Plugin types
    const plugins = this.getByConstruct('plugin')
    if (plugins.length > 0) {
      sections.push('\n### Plugin Types (for PLUGIN template declarations)')
      for (const p of plugins) {
        const requiredFields = p.fields.filter(f => f.required).map(f => f.key)
        let line = `- **${p.type_name}**: ${p.nl_description}`
        if (requiredFields.length > 0) {
          line += `\n  Required fields: ${requiredFields.join(', ')}`
        }
        if (p.requires_backend && p.requires_backend.length > 0) {
          line += `\n  Requires backend: ${p.requires_backend.join(', ')}`
        }
        sections.push(line)
      }
    }

    // Algorithm types
    const algorithms = this.getByConstruct('algorithm')
    if (algorithms.length > 0) {
      sections.push('\n### Algorithm Types (for ROUTE ALGORITHM selection)')
      for (const a of algorithms) {
        const fields = a.fields.map(f => `${f.key}(${f.type})`).join(', ')
        let line = `- **${a.type_name}**: ${a.nl_description}`
        if (fields) {
          line += `\n  Params: ${fields}`
        }
        if (a.requires_signal && a.requires_signal.length > 0) {
          line += `\n  Requires signal: ${a.requires_signal.join(', ')}`
        }
        sections.push(line)
      }
    }

    // Backend types
    const backends = this.getByConstruct('backend')
    if (backends.length > 0) {
      sections.push('\n### Backend Types (for BACKEND declarations)')
      for (const b of backends) {
        sections.push(`- **${b.type_name}**: ${b.nl_description}`)
      }
    }

    return sections.join('\n')
  }
}

// ─────────────────────────────────────────────
// Default Registry — pre-populated with all known types
// ─────────────────────────────────────────────

function createDefaultRegistry(): NLSchemaRegistry {
  const registry = new NLSchemaRegistry()

  // ── Signal Types ──
  const signalEntries: NLSchemaEntry[] = [
    {
      construct: 'signal',
      type_name: 'keyword',
      nl_triggers: ['keyword', 'keywords', 'pattern', 'patterns', 'match', 'matching', 'contains', 'word', 'regex', 'text match', '关键词', '匹配', '包含'],
      nl_description: 'Matches queries containing specific keywords or patterns using regex, BM25, or n-gram methods',
      nl_examples: [
        'Match queries containing "urgent" or "emergency"',
        'Detect when users mention specific product names',
        'Route requests that include programming keywords',
      ],
      fields: getSignalFieldSchema('keyword'),
    },
    {
      construct: 'signal',
      type_name: 'embedding',
      nl_triggers: ['embedding', 'semantic', 'similarity', 'vector', 'cosine', 'similar to', 'related to', '语义', '相似', '向量'],
      nl_description: 'Measures semantic similarity between query and candidate phrases using embedding vectors',
      nl_examples: [
        'Route queries semantically similar to "calculus" and "algebra"',
        'Detect queries about cooking recipes using semantic matching',
      ],
      fields: getSignalFieldSchema('embedding'),
    },
    {
      construct: 'signal',
      type_name: 'domain',
      nl_triggers: ['domain', 'topic', 'category', 'subject', 'field', 'area', 'about', '领域', '主题', '分类', '话题'],
      nl_description: 'Classifies queries into topic domains (e.g., math, coding, medical) using MMLU-style categorization',
      nl_examples: [
        'Route math questions to the reasoning model',
        'Detect coding-related queries',
        'Classify queries about medical topics',
      ],
      fields: getSignalFieldSchema('domain'),
    },
    {
      construct: 'signal',
      type_name: 'fact_check',
      nl_triggers: ['fact check', 'factual', 'verify', 'verification', 'accuracy', 'truth', '事实', '验证', '核查'],
      nl_description: 'Detects queries that require factual verification',
      nl_examples: [
        'Flag queries that need fact-checking',
        'Route fact-verification requests to a specialized model',
      ],
      fields: getSignalFieldSchema('fact_check'),
    },
    {
      construct: 'signal',
      type_name: 'user_feedback',
      nl_triggers: ['feedback', 'rating', 'user feedback', 'score', 'satisfaction', '反馈', '评分', '评价'],
      nl_description: 'Captures user feedback/ratings for model performance evaluation',
      nl_examples: [
        'Use user feedback to drive RL-based model selection',
        'Collect ratings for Elo algorithm',
      ],
      fields: getSignalFieldSchema('user_feedback'),
      requires_signal: [],
    },
    {
      construct: 'signal',
      type_name: 'preference',
      nl_triggers: ['preference', 'prefer', 'favorite', 'choice', '偏好', '首选'],
      nl_description: 'Captures user preferences for routing decisions',
      nl_examples: [
        'Route based on user model preference',
      ],
      fields: getSignalFieldSchema('preference'),
    },
    {
      construct: 'signal',
      type_name: 'language',
      nl_triggers: ['language', 'locale', 'lang', 'chinese', 'english', 'multilingual', 'translation', '语言', '多语言', '翻译'],
      nl_description: 'Detects the language of the query',
      nl_examples: [
        'Route Chinese queries to a Chinese-optimized model',
        'Detect query language for multilingual routing',
      ],
      fields: getSignalFieldSchema('language'),
    },
    {
      construct: 'signal',
      type_name: 'context',
      nl_triggers: ['context', 'token', 'tokens', 'length', 'long', 'short', 'context window', 'context length', '上下文', '长度', '令牌'],
      nl_description: 'Measures the token count of the query to route based on context length',
      nl_examples: [
        'Route long queries (>8K tokens) to models with large context windows',
        'Use lightweight models for short queries',
      ],
      fields: getSignalFieldSchema('context'),
    },
    {
      construct: 'signal',
      type_name: 'complexity',
      nl_triggers: ['complexity', 'complex', 'difficult', 'hard', 'easy', 'simple', 'difficulty', '复杂', '难度', '简单', '困难'],
      nl_description: 'Evaluates the difficulty/complexity of queries using classifier or contrastive methods',
      nl_examples: [
        'Route complex queries to powerful models, simple ones to lightweight models',
        'Classify query difficulty for cost-aware routing',
      ],
      fields: getSignalFieldSchema('complexity'),
    },
    {
      construct: 'signal',
      type_name: 'modality',
      nl_triggers: ['modality', 'image', 'audio', 'video', 'multimodal', 'vision', 'text', '模态', '图片', '图像', '音频', '多模态'],
      nl_description: 'Detects the input modality (text, image, audio, video)',
      nl_examples: [
        'Route image-related queries to a vision model',
        'Detect multimodal inputs',
      ],
      fields: getSignalFieldSchema('modality'),
    },
    {
      construct: 'signal',
      type_name: 'authz',
      nl_triggers: ['auth', 'authorization', 'permission', 'role', 'access', 'rbac', 'group', 'user group', '授权', '权限', '角色'],
      nl_description: 'Checks user authorization and role-based access for premium/tiered routing',
      nl_examples: [
        'Route premium users to the best model',
        'Restrict access to GPT-4o for admin users only',
      ],
      fields: getSignalFieldSchema('authz'),
    },
    {
      construct: 'signal',
      type_name: 'jailbreak',
      nl_triggers: ['jailbreak', 'injection', 'prompt injection', 'attack', 'adversarial', 'safety', '越狱', '注入', '攻击', '安全'],
      nl_description: 'Detects jailbreak/prompt injection attempts using classifier or contrastive methods',
      nl_examples: [
        'Block jailbreak attempts before they reach the model',
        'Add jailbreak detection as a guardrail',
      ],
      fields: getSignalFieldSchema('jailbreak'),
    },
    {
      construct: 'signal',
      type_name: 'pii',
      nl_triggers: ['pii', 'personal', 'sensitive', 'privacy', 'mask', 'redact', 'email', 'phone', 'data protection', '隐私', '敏感', '脱敏', '个人信息'],
      nl_description: 'Detects personally identifiable information (PII) in queries for masking or blocking',
      nl_examples: [
        'Mask PII before sending to the model',
        'Add PII protection to all routes',
        'Block requests containing phone numbers or emails',
      ],
      fields: getSignalFieldSchema('pii'),
    },
  ]

  // ── Plugin Types ──
  const pluginEntries: NLSchemaEntry[] = [
    {
      construct: 'plugin',
      type_name: 'semantic_cache',
      nl_triggers: ['cache', 'caching', 'cached', 'semantic cache', 'response cache', '缓存', '语义缓存'],
      nl_description: PLUGIN_DESCRIPTIONS.semantic_cache,
      nl_examples: [
        'Add response caching with 0.95 similarity threshold',
        'Enable semantic caching to reduce latency',
      ],
      fields: getPluginFieldSchema('semantic_cache'),
      requires_backend: ['semantic_cache'],
    },
    {
      construct: 'plugin',
      type_name: 'memory',
      nl_triggers: ['memory', 'conversation memory', 'remember', 'history', 'recall', '记忆', '对话记忆', '历史'],
      nl_description: PLUGIN_DESCRIPTIONS.memory,
      nl_examples: [
        'Add conversation memory that remembers the last 5 interactions',
        'Enable persistent memory with vector retrieval',
      ],
      fields: getPluginFieldSchema('memory'),
      requires_backend: ['memory'],
    },
    {
      construct: 'plugin',
      type_name: 'system_prompt',
      nl_triggers: ['system prompt', 'system message', 'persona', 'instruction', 'role', 'character', '系统提示', '人设', '角色设定'],
      nl_description: PLUGIN_DESCRIPTIONS.system_prompt,
      nl_examples: [
        'Set the system prompt to "You are a helpful math tutor"',
        'Replace the system prompt for coding routes',
      ],
      fields: getPluginFieldSchema('system_prompt'),
    },
    {
      construct: 'plugin',
      type_name: 'hallucination',
      nl_triggers: ['hallucination', 'hallucinate', 'fact check', 'verify output', 'nli', 'grounding', '幻觉', '幻觉检测'],
      nl_description: PLUGIN_DESCRIPTIONS.hallucination,
      nl_examples: [
        'Add hallucination detection using NLI',
        'Verify model output for factual accuracy',
      ],
      fields: getPluginFieldSchema('hallucination'),
    },
    {
      construct: 'plugin',
      type_name: 'rag',
      nl_triggers: ['rag', 'retrieval', 'retrieval augmented', 'knowledge base', 'document', 'search', '检索', '知识库', '文档检索'],
      nl_description: PLUGIN_DESCRIPTIONS.rag,
      nl_examples: [
        'Add RAG with top-5 document retrieval',
        'Inject retrieved context into the system prompt',
      ],
      fields: getPluginFieldSchema('rag'),
      requires_backend: ['vector_store'],
    },
    {
      construct: 'plugin',
      type_name: 'header_mutation',
      nl_triggers: ['header', 'http header', 'custom header', 'add header', 'request header', '请求头', 'HTTP头'],
      nl_description: PLUGIN_DESCRIPTIONS.header_mutation,
      nl_examples: [
        'Add a custom X-Route-Info header',
        'Remove the Authorization header before forwarding',
      ],
      fields: getPluginFieldSchema('header_mutation'),
    },
    {
      construct: 'plugin',
      type_name: 'router_replay',
      nl_triggers: ['replay', 'record', 'debug', 'trace', 'capture', 'logging', '回放', '录制', '调试', '追踪'],
      nl_description: PLUGIN_DESCRIPTIONS.router_replay,
      nl_examples: [
        'Enable request/response recording for debugging',
        'Capture up to 10000 request-response pairs',
      ],
      fields: getPluginFieldSchema('router_replay'),
    },
    {
      construct: 'plugin',
      type_name: 'image_gen',
      nl_triggers: ['image generation', 'image gen', 'generate image', 'dall-e', 'stable diffusion', 'text to image', '图片生成', '文生图'],
      nl_description: PLUGIN_DESCRIPTIONS.image_gen,
      nl_examples: [
        'Route image generation requests to the DALL-E backend',
      ],
      fields: getPluginFieldSchema('image_gen'),
      requires_backend: ['image_gen_backend'],
    },
    {
      construct: 'plugin',
      type_name: 'fast_response',
      nl_triggers: ['fast response', 'quick reply', 'block', 'reject', 'deny', 'refuse', 'short circuit', '快速响应', '拒绝', '拦截'],
      nl_description: PLUGIN_DESCRIPTIONS.fast_response,
      nl_examples: [
        'Block harmful requests with a rejection message',
        'Return a quick response without calling upstream models',
      ],
      fields: getPluginFieldSchema('fast_response'),
    },
  ]

  // ── Algorithm Types ──
  const algorithmEntries: NLSchemaEntry[] = [
    {
      construct: 'algorithm',
      type_name: 'confidence',
      nl_triggers: ['confidence', 'cascade', 'escalate', 'try smaller first', 'logprob', '置信度', '级联', '逐级升级'],
      nl_description: ALGORITHM_DESCRIPTIONS.confidence,
      nl_examples: [
        'Use confidence-based cascade: try GPT-4o-mini first, escalate to GPT-4o',
        'Set confidence threshold to 0.8',
      ],
      fields: getAlgorithmFieldSchema('confidence'),
    },
    {
      construct: 'algorithm',
      type_name: 'ratings',
      nl_triggers: ['ratings', 'compare', 'side by side', 'concurrent', 'multiple choices', '评分', '对比', '并行'],
      nl_description: ALGORITHM_DESCRIPTIONS.ratings,
      nl_examples: [
        'Run all models concurrently and return multiple choices',
        'Use ratings to compare model outputs',
      ],
      fields: getAlgorithmFieldSchema('ratings'),
    },
    {
      construct: 'algorithm',
      type_name: 'remom',
      nl_triggers: ['remom', 'reasoning mixture', 'multi-round', 'parallel reasoning', 'synthesis', '推理混合', '多轮推理'],
      nl_description: ALGORITHM_DESCRIPTIONS.remom,
      nl_examples: [
        'Use ReMoM with breadth schedule [32, 4]',
        'Enable multi-round parallel reasoning with intelligent synthesis',
      ],
      fields: getAlgorithmFieldSchema('remom'),
    },
    {
      construct: 'algorithm',
      type_name: 'static',
      nl_triggers: ['static', 'fixed', 'no algorithm', 'direct', '静态', '固定'],
      nl_description: ALGORITHM_DESCRIPTIONS.static,
      nl_examples: [
        'Use static routing with fixed scores',
        'Route directly without any algorithm',
      ],
      fields: getAlgorithmFieldSchema('static'),
    },
    {
      construct: 'algorithm',
      type_name: 'elo',
      nl_triggers: ['elo', 'elo rating', 'bradley-terry', 'ranking', '排名', 'Elo评分'],
      nl_description: ALGORITHM_DESCRIPTIONS.elo,
      nl_examples: [
        'Use Elo rating system with initial rating 1500',
        'Enable per-category Elo for model selection',
      ],
      fields: getAlgorithmFieldSchema('elo'),
      requires_signal: ['user_feedback'],
    },
    {
      construct: 'algorithm',
      type_name: 'router_dc',
      nl_triggers: ['router dc', 'contrastive', 'dual contrastive', 'contrastive learning', '对比学习'],
      nl_description: ALGORITHM_DESCRIPTIONS.router_dc,
      nl_examples: [
        'Use dual-contrastive learning for query-model matching',
      ],
      fields: getAlgorithmFieldSchema('router_dc'),
    },
    {
      construct: 'algorithm',
      type_name: 'automix',
      nl_triggers: ['automix', 'cost aware', 'cost quality', 'pomdp', 'cost optimization', '成本优化', '成本感知'],
      nl_description: ALGORITHM_DESCRIPTIONS.automix,
      nl_examples: [
        'Use AutoMix for cost-quality optimization',
        'Enable cost-aware routing with POMDP',
      ],
      fields: getAlgorithmFieldSchema('automix'),
    },
    {
      construct: 'algorithm',
      type_name: 'hybrid',
      nl_triggers: ['hybrid', 'combined', 'weighted', 'ensemble', 'mix', '混合', '组合', '加权'],
      nl_description: ALGORITHM_DESCRIPTIONS.hybrid,
      nl_examples: [
        'Combine Elo, RouterDC, and AutoMix with custom weights',
      ],
      fields: getAlgorithmFieldSchema('hybrid'),
      requires_signal: ['user_feedback'],
    },
    {
      construct: 'algorithm',
      type_name: 'rl_driven',
      nl_triggers: ['rl', 'reinforcement learning', 'thompson sampling', 'exploration', 'bandit', '强化学习', '探索'],
      nl_description: ALGORITHM_DESCRIPTIONS.rl_driven,
      nl_examples: [
        'Use reinforcement learning with Thompson Sampling',
        'Enable RL-driven model selection with personalization',
      ],
      fields: getAlgorithmFieldSchema('rl_driven'),
      requires_signal: ['user_feedback'],
    },
    {
      construct: 'algorithm',
      type_name: 'gmtrouter',
      nl_triggers: ['gmtrouter', 'graph', 'heterogeneous graph', 'personalized routing', '图路由', '个性化路由'],
      nl_description: ALGORITHM_DESCRIPTIONS.gmtrouter,
      nl_examples: [
        'Use GMT router with personalized graph learning',
      ],
      fields: getAlgorithmFieldSchema('gmtrouter'),
      requires_signal: ['user_feedback'],
    },
    {
      construct: 'algorithm',
      type_name: 'latency_aware',
      nl_triggers: ['latency', 'latency aware', 'fast', 'speed', 'tpot', 'ttft', 'response time', '延迟', '速度', '响应时间'],
      nl_description: ALGORITHM_DESCRIPTIONS.latency_aware,
      nl_examples: [
        'Select models based on TPOT and TTFT percentiles',
        'Optimize for low latency',
      ],
      fields: getAlgorithmFieldSchema('latency_aware'),
    },
    {
      construct: 'algorithm',
      type_name: 'knn',
      nl_triggers: ['knn', 'k nearest', 'nearest neighbor', 'KNN'],
      nl_description: ALGORITHM_DESCRIPTIONS.knn,
      nl_examples: [
        'Use KNN for query-based model selection',
      ],
      fields: getAlgorithmFieldSchema('knn'),
    },
    {
      construct: 'algorithm',
      type_name: 'kmeans',
      nl_triggers: ['kmeans', 'k-means', 'clustering', '聚类'],
      nl_description: ALGORITHM_DESCRIPTIONS.kmeans,
      nl_examples: [
        'Use KMeans clustering for model selection',
      ],
      fields: getAlgorithmFieldSchema('kmeans'),
    },
    {
      construct: 'algorithm',
      type_name: 'svm',
      nl_triggers: ['svm', 'support vector', 'classification', 'SVM'],
      nl_description: ALGORITHM_DESCRIPTIONS.svm,
      nl_examples: [
        'Use SVM for model classification',
      ],
      fields: getAlgorithmFieldSchema('svm'),
    },
  ]

  // ── Backend Types ──
  // Note: Backend types don't have getBackendFieldSchema() yet — using generic fields
  const backendEntries: NLSchemaEntry[] = [
    {
      construct: 'backend',
      type_name: 'vllm_endpoint',
      nl_triggers: ['vllm', 'endpoint', 'model server', 'inference server', 'model endpoint', '推理服务', '模型端点'],
      nl_description: 'vLLM-compatible model inference endpoint',
      nl_examples: [
        'Add a vLLM endpoint at localhost:8000',
      ],
      fields: [
        { key: 'host', label: 'Host', type: 'string', required: true, placeholder: 'localhost' },
        { key: 'port', label: 'Port', type: 'number', required: true, placeholder: '8000' },
        { key: 'model_name', label: 'Model Name', type: 'string', placeholder: 'deepseek-r1' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'provider_profile',
      nl_triggers: ['provider', 'api', 'openai', 'anthropic', 'api key', 'api provider', 'cloud provider', '提供商', '云服务'],
      nl_description: 'API provider profile (OpenAI, Anthropic, etc.) with authentication',
      nl_examples: [
        'Add an OpenAI provider with API key',
        'Configure Anthropic as a backend provider',
      ],
      fields: [
        { key: 'provider', label: 'Provider', type: 'string', required: true, placeholder: 'openai' },
        { key: 'api_key_from', label: 'API Key From', type: 'string', placeholder: 'env:OPENAI_API_KEY' },
        { key: 'base_url', label: 'Base URL', type: 'string', placeholder: 'https://api.openai.com/v1' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'embedding_model',
      nl_triggers: ['embedding model', 'embedding backend', 'embedding server', 'embeddings', '嵌入模型'],
      nl_description: 'Embedding model backend for semantic similarity computation',
      nl_examples: [
        'Add an embedding model backend for semantic signals',
      ],
      fields: [
        { key: 'model_name', label: 'Model Name', type: 'string', required: true, placeholder: 'text-embedding-3-small' },
        { key: 'host', label: 'Host', type: 'string' },
        { key: 'port', label: 'Port', type: 'number' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'semantic_cache',
      nl_triggers: ['cache backend', 'cache store', 'cache storage', '缓存后端', '缓存存储'],
      nl_description: 'Storage backend for semantic cache plugin',
      nl_examples: [
        'Add a semantic cache backend',
      ],
      fields: [
        { key: 'host', label: 'Host', type: 'string' },
        { key: 'port', label: 'Port', type: 'number' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'memory',
      nl_triggers: ['memory backend', 'memory store', 'memory storage', '记忆后端', '记忆存储'],
      nl_description: 'Storage backend for conversation memory plugin',
      nl_examples: [
        'Add a memory storage backend',
      ],
      fields: [
        { key: 'host', label: 'Host', type: 'string' },
        { key: 'port', label: 'Port', type: 'number' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'response_api',
      nl_triggers: ['response api', 'responses api', 'openai responses', '响应API'],
      nl_description: 'OpenAI Responses API endpoint',
      nl_examples: [
        'Add a Responses API backend',
      ],
      fields: [
        { key: 'base_url', label: 'Base URL', type: 'string' },
      ],
    },
    {
      construct: 'backend',
      type_name: 'vector_store',
      nl_triggers: ['vector store', 'vector db', 'vector database', 'pinecone', 'weaviate', 'milvus', 'chromadb', '向量数据库', '向量存储'],
      nl_description: 'Vector database backend for RAG plugin document retrieval',
      nl_examples: [
        'Add a vector store backend for RAG',
      ],
      fields: [
        { key: 'host', label: 'Host', type: 'string', required: true },
        { key: 'port', label: 'Port', type: 'number' },
        { key: 'collection', label: 'Collection', type: 'string' },
      ],
      requires_backend: [],
    },
    {
      construct: 'backend',
      type_name: 'image_gen_backend',
      nl_triggers: ['image gen backend', 'image generation backend', 'dall-e backend', '图片生成后端'],
      nl_description: 'Backend for image generation services',
      nl_examples: [
        'Add an image generation backend',
      ],
      fields: [
        { key: 'host', label: 'Host', type: 'string', required: true },
        { key: 'port', label: 'Port', type: 'number' },
        { key: 'model_name', label: 'Model Name', type: 'string' },
      ],
    },
  ]

  registry.registerAll(signalEntries)
  registry.registerAll(pluginEntries)
  registry.registerAll(algorithmEntries)
  registry.registerAll(backendEntries)

  return registry
}

/** Singleton default registry instance with all built-in types pre-registered */
export const defaultRegistry = createDefaultRegistry()
