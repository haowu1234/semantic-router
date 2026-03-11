/**
 * NL Semantic Cache
 *
 * Caches NL → DSL generation results to avoid redundant LLM calls.
 * Uses a combination of exact matching and fuzzy matching for cache lookup.
 *
 * Features:
 * - Exact match on normalized input
 * - Fuzzy match using Jaccard similarity (for near-duplicates)
 * - LRU eviction policy
 * - TTL-based expiration
 * - LocalStorage persistence for cross-session caching
 */

import type { NLGenerateResult, NLContext } from './nlPipeline'

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

const CACHE_MAX_SIZE = 100
const CACHE_TTL_MS = 24 * 60 * 60 * 1000 // 24 hours
const FUZZY_THRESHOLD = 0.85 // Jaccard similarity threshold
const STORAGE_KEY = 'nl_cache_v1'

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface CacheEntry {
  /** Normalized input text */
  normalizedInput: string
  /** Operation mode (generate/modify) */
  mode: 'generate' | 'modify'
  /** Hash of current DSL (for modify mode) */
  dslHash: string | null
  /** The cached result */
  result: NLGenerateResult
  /** Timestamp when cached */
  timestamp: number
  /** Hit count for LRU */
  hits: number
  /** Input tokens for fuzzy matching */
  tokens: string[]
}

interface CacheStats {
  size: number
  hits: number
  misses: number
  fuzzyHits: number
}

// ─────────────────────────────────────────────
// NLCache Class
// ─────────────────────────────────────────────

export class NLCache {
  private entries: Map<string, CacheEntry> = new Map()
  private stats: CacheStats = { size: 0, hits: 0, misses: 0, fuzzyHits: 0 }

  constructor() {
    this.loadFromStorage()
  }

  /**
   * Get a cached result for the given input and context.
   * Returns null if no suitable cache entry exists.
   */
  get(input: string, context: NLContext): NLGenerateResult | null {
    const normalized = this.normalizeInput(input)
    const mode = this.detectMode(context)
    const dslHash = context.currentDSL ? this.hashString(context.currentDSL) : null

    // 1. Try exact match
    const exactKey = this.buildKey(normalized, mode, dslHash)
    const exactEntry = this.entries.get(exactKey)

    if (exactEntry && !this.isExpired(exactEntry)) {
      exactEntry.hits++
      this.stats.hits++
      return exactEntry.result
    }

    // 2. Try fuzzy match (only for generate mode to avoid stale modify results)
    if (mode === 'generate') {
      const fuzzyMatch = this.findFuzzyMatch(normalized, mode)
      if (fuzzyMatch) {
        fuzzyMatch.hits++
        this.stats.hits++
        this.stats.fuzzyHits++
        return fuzzyMatch.result
      }
    }

    this.stats.misses++
    return null
  }

  /**
   * Store a result in the cache.
   */
  set(input: string, context: NLContext, result: NLGenerateResult): void {
    // Don't cache invalid results
    if (!result.isValid) return

    const normalized = this.normalizeInput(input)
    const mode = this.detectMode(context)
    const dslHash = context.currentDSL ? this.hashString(context.currentDSL) : null
    const key = this.buildKey(normalized, mode, dslHash)
    const tokens = this.tokenize(normalized)

    const entry: CacheEntry = {
      normalizedInput: normalized,
      mode,
      dslHash,
      result,
      timestamp: Date.now(),
      hits: 0,
      tokens,
    }

    // Evict if at capacity
    if (this.entries.size >= CACHE_MAX_SIZE) {
      this.evictLRU()
    }

    this.entries.set(key, entry)
    this.stats.size = this.entries.size
    this.saveToStorage()
  }

  /**
   * Clear all cached entries.
   */
  clear(): void {
    this.entries.clear()
    this.stats = { size: 0, hits: 0, misses: 0, fuzzyHits: 0 }
    this.saveToStorage()
  }

  /**
   * Get cache statistics.
   */
  getStats(): CacheStats {
    return { ...this.stats }
  }

  /**
   * Invalidate cache entries that match the given DSL hash.
   * Called when the DSL is modified externally.
   */
  invalidateByDSL(dsl: string): void {
    const hash = this.hashString(dsl)
    for (const [key, entry] of this.entries) {
      if (entry.dslHash === hash) {
        this.entries.delete(key)
      }
    }
    this.stats.size = this.entries.size
    this.saveToStorage()
  }

  // ─────────────────────────────────────────────
  // Private Methods
  // ─────────────────────────────────────────────

  private normalizeInput(input: string): string {
    return input
      .toLowerCase()
      .trim()
      .replace(/\s+/g, ' ')
      .replace(/[^\w\s]/g, '') // Remove punctuation for better matching
  }

  private tokenize(input: string): string[] {
    return input.split(/\s+/).filter(t => t.length > 2)
  }

  private buildKey(normalized: string, mode: string, dslHash: string | null): string {
    return `${mode}:${dslHash || 'null'}:${normalized}`
  }

  private detectMode(context: NLContext): 'generate' | 'modify' {
    return context.currentDSL?.trim() ? 'modify' : 'generate'
  }

  private hashString(str: string): string {
    // Simple hash for cache key purposes (not cryptographic)
    let hash = 0
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return hash.toString(36)
  }

  private isExpired(entry: CacheEntry): boolean {
    return Date.now() - entry.timestamp > CACHE_TTL_MS
  }

  private findFuzzyMatch(normalized: string, mode: string): CacheEntry | null {
    const inputTokens = new Set(this.tokenize(normalized))
    let bestMatch: CacheEntry | null = null
    let bestSimilarity = 0

    for (const entry of this.entries.values()) {
      // Skip expired entries
      if (this.isExpired(entry)) continue

      // Skip different modes
      if (entry.mode !== mode) continue

      // Calculate Jaccard similarity
      const entryTokens = new Set(entry.tokens)
      const similarity = this.jaccardSimilarity(inputTokens, entryTokens)

      if (similarity >= FUZZY_THRESHOLD && similarity > bestSimilarity) {
        bestMatch = entry
        bestSimilarity = similarity
      }
    }

    return bestMatch
  }

  private jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
    if (setA.size === 0 && setB.size === 0) return 1
    if (setA.size === 0 || setB.size === 0) return 0

    let intersection = 0
    for (const item of setA) {
      if (setB.has(item)) intersection++
    }

    const union = setA.size + setB.size - intersection
    return intersection / union
  }

  private evictLRU(): void {
    // Find the least recently used entry (lowest hits, oldest timestamp)
    let lruKey: string | null = null
    let lruScore = Infinity

    for (const [key, entry] of this.entries) {
      // Expired entries get priority for eviction
      if (this.isExpired(entry)) {
        this.entries.delete(key)
        continue
      }

      // Score = hits - age_penalty
      const agePenalty = (Date.now() - entry.timestamp) / CACHE_TTL_MS
      const score = entry.hits - agePenalty

      if (score < lruScore) {
        lruScore = score
        lruKey = key
      }
    }

    if (lruKey) {
      this.entries.delete(lruKey)
    }
  }

  private saveToStorage(): void {
    try {
      const data = {
        entries: Array.from(this.entries.entries()),
        stats: this.stats,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
    } catch {
      // Storage might be full or disabled
    }
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (!stored) return

      const data = JSON.parse(stored)

      // Restore entries
      if (Array.isArray(data.entries)) {
        for (const [key, entry] of data.entries) {
          // Skip expired entries during load
          if (!this.isExpired(entry)) {
            this.entries.set(key, entry)
          }
        }
      }

      // Restore stats
      if (data.stats) {
        this.stats = {
          size: this.entries.size,
          hits: data.stats.hits || 0,
          misses: data.stats.misses || 0,
          fuzzyHits: data.stats.fuzzyHits || 0,
        }
      }
    } catch {
      // Invalid data in storage, start fresh
    }
  }
}

// ─────────────────────────────────────────────
// Singleton Instance
// ─────────────────────────────────────────────

export const nlCache = new NLCache()

// ─────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────

/**
 * Clear the NL cache (useful for debugging or reset).
 */
export function clearNLCache(): void {
  nlCache.clear()
}

/**
 * Get cache statistics for debugging.
 */
export function getNLCacheStats(): CacheStats {
  return nlCache.getStats()
}
