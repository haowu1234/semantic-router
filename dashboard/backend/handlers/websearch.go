package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// ========================
// Error codes for better user feedback
// ========================

type SearchErrorCode string

const (
	ErrCodeInvalidRequest SearchErrorCode = "INVALID_REQUEST"
	ErrCodeEmptyQuery     SearchErrorCode = "EMPTY_QUERY"
	ErrCodeQueryTooLong   SearchErrorCode = "QUERY_TOO_LONG"
	ErrCodeRateLimited    SearchErrorCode = "RATE_LIMITED"
	ErrCodeSearchFailed   SearchErrorCode = "SEARCH_FAILED"
	ErrCodeTimeout        SearchErrorCode = "TIMEOUT"
	ErrCodeUpstreamError  SearchErrorCode = "UPSTREAM_ERROR"
	ErrCodeInvalidURL     SearchErrorCode = "INVALID_URL"
)

// Error messages in both English and Chinese for better UX
var errorMessages = map[SearchErrorCode]string{
	ErrCodeInvalidRequest: "Invalid request body / 无效的请求体",
	ErrCodeEmptyQuery:     "Search query is required / 搜索查询不能为空",
	ErrCodeQueryTooLong:   "Query too long (max 500 chars) / 查询过长（最多500字符）",
	ErrCodeRateLimited:    "Search service is busy, please try again in a moment / 搜索服务繁忙，请稍后再试",
	ErrCodeSearchFailed:   "Search failed / 搜索失败",
	ErrCodeTimeout:        "Search timeout, please try again / 搜索超时，请重试",
	ErrCodeUpstreamError:  "Search service temporarily unavailable / 搜索服务暂时不可用",
	ErrCodeInvalidURL:     "Invalid URL detected / 检测到无效URL",
}

// ========================
// Configuration constants
// ========================

const (
	maxQueryLength     = 500 // Maximum query length
	maxResultsLimit    = 10  // Maximum results per request
	defaultNumResults  = 5   // Default number of results
	httpTimeout        = 15 * time.Second
	maxRetries         = 3 // Retry attempts for transient failures
	retryBaseDelay     = 1 * time.Second
	rateLimitWindow    = time.Minute // Rate limit window
	rateLimitMaxReqs   = 5           // Max requests per window per IP (strict for public service)
	globalRateLimit    = 30          // Global max requests per window (0.5 req/sec to avoid ban)
	maxConcurrent      = 2           // Max concurrent outgoing requests (very conservative)
	maxTrackedIPs      = 10000       // Max IPs to track (memory protection)
	minRequestInterval = 1 * time.Second
	maxRequestInterval = 3 * time.Second
)

// ========================
// Pre-compiled regex patterns (performance optimization)
// ========================

var (
	// Pattern to match search result links
	linkPattern = regexp.MustCompile(`<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>`)
	// Pattern to match snippets
	snippetPattern = regexp.MustCompile(`<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*</[^>]*>)*[^<]*)</a>`)
	// Alternative snippet pattern
	snippetAltPattern = regexp.MustCompile(`class="result__snippet"[^>]*>([^<]+)`)
	// Pattern for uddg URL extraction
	uddgPattern = regexp.MustCompile(`uddg=([^&"]+)`)
	// Pattern for title extraction in fallback
	titlePattern = regexp.MustCompile(`result__a[^>]*>([^<]+)`)
	// Pattern for snippet extraction in fallback
	snippetFallbackPattern = regexp.MustCompile(`result__snippet[^>]*>([^<]+)`)
	// Pattern for HTML tag removal
	htmlTagPattern = regexp.MustCompile(`<[^>]*>`)
	// Pattern for whitespace normalization
	whitespacePattern = regexp.MustCompile(`\s+`)
)

// User-Agent rotation to reduce detection risk
var userAgents = []string{
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
	"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
	"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
	"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
}

// getRandomUserAgent returns a random User-Agent string
func getRandomUserAgent() string {
	return userAgents[rand.Intn(len(userAgents))]
}

// randomDelay adds a random delay between min and max intervals to avoid detection
func randomDelay() {
	delay := minRequestInterval + time.Duration(rand.Int63n(int64(maxRequestInterval-minRequestInterval)))
	time.Sleep(delay)
}

// ========================
// HTTP Client pool (connection reuse)
// ========================

var (
	httpClient     *http.Client
	httpClientOnce sync.Once
)

// getHTTPClient returns a shared HTTP client with connection pooling
func getHTTPClient() *http.Client {
	httpClientOnce.Do(func() {
		transport := &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
		}
		httpClient = &http.Client{
			Timeout:   httpTimeout,
			Transport: transport,
		}
	})
	return httpClient
}

// ========================
// Rate limiter with global limit
// ========================

type rateLimiter struct {
	mu         sync.RWMutex
	requests   map[string][]time.Time
	globalReqs []time.Time // Track all requests globally
}

var globalRateLimiter = &rateLimiter{
	requests:   make(map[string][]time.Time),
	globalReqs: make([]time.Time, 0),
}

// Semaphore for concurrent request limiting
var concurrentSem = make(chan struct{}, maxConcurrent)

// isAllowed checks if a request from the given IP is allowed
func (rl *rateLimiter) isAllowed(ip string) (bool, SearchErrorCode) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	windowStart := now.Add(-rateLimitWindow)

	// Check global rate limit first
	var recentGlobal []time.Time
	for _, t := range rl.globalReqs {
		if t.After(windowStart) {
			recentGlobal = append(recentGlobal, t)
		}
	}
	if len(recentGlobal) >= globalRateLimit {
		rl.globalReqs = recentGlobal
		return false, ErrCodeRateLimited
	}

	// Memory protection: limit tracked IPs
	if len(rl.requests) >= maxTrackedIPs {
		if _, exists := rl.requests[ip]; !exists {
			return false, ErrCodeRateLimited
		}
	}

	// Check per-IP rate limit
	var recentReqs []time.Time
	for _, t := range rl.requests[ip] {
		if t.After(windowStart) {
			recentReqs = append(recentReqs, t)
		}
	}

	if len(recentReqs) >= rateLimitMaxReqs {
		rl.requests[ip] = recentReqs
		return false, ErrCodeRateLimited
	}

	// Record the request
	rl.requests[ip] = append(recentReqs, now)
	recentGlobal = append(recentGlobal, now)
	rl.globalReqs = recentGlobal
	return true, ""
}

// cleanup removes stale entries (call periodically)
func (rl *rateLimiter) cleanup() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	windowStart := time.Now().Add(-rateLimitWindow)

	// Clean per-IP entries
	for ip, times := range rl.requests {
		var valid []time.Time
		for _, t := range times {
			if t.After(windowStart) {
				valid = append(valid, t)
			}
		}
		if len(valid) == 0 {
			delete(rl.requests, ip)
		} else {
			rl.requests[ip] = valid
		}
	}

	// Clean global entries
	var validGlobal []time.Time
	for _, t := range rl.globalReqs {
		if t.After(windowStart) {
			validGlobal = append(validGlobal, t)
		}
	}
	rl.globalReqs = validGlobal
}

// getStats returns current rate limiter statistics (for monitoring/debugging)
// nolint:unused // Reserved for future monitoring endpoint
func (rl *rateLimiter) getStats() (trackedIPs int, globalReqCount int) {
	rl.mu.RLock()
	defer rl.mu.RUnlock()
	return len(rl.requests), len(rl.globalReqs)
}

// Start cleanup goroutine
func init() {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		for range ticker.C {
			globalRateLimiter.cleanup()
		}
	}()
}

// ========================
// Data structures
// ========================

// SearchResult represents a single search result
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
	Domain  string `json:"domain"`
}

// WebSearchRequest represents the incoming search request
type WebSearchRequest struct {
	Query      string `json:"query"`
	NumResults int    `json:"num_results,omitempty"`
}

// WebSearchResponse represents the search response
type WebSearchResponse struct {
	Query   string          `json:"query"`
	Results []SearchResult  `json:"results"`
	Error   string          `json:"error,omitempty"`
	Code    SearchErrorCode `json:"code,omitempty"`
	Backend string          `json:"backend,omitempty"` // Which backend was used
}

// ========================
// Search Backend Interface
// ========================

// SearchBackend defines the interface for different search providers
type SearchBackend interface {
	Search(query string, numResults int) ([]SearchResult, error)
	Name() string
}

// SearchManager manages multiple search backends with fallback support
type SearchManager struct {
	backends []SearchBackend
	mu       sync.RWMutex
}

var globalSearchManager *SearchManager
var searchManagerOnce sync.Once

// InitSearchManager initializes the search manager with configured backends
func InitSearchManager(cfg *config.Config) {
	searchManagerOnce.Do(func() {
		globalSearchManager = &SearchManager{
			backends: buildBackends(cfg),
		}
		log.Printf("Search backends initialized: %v", getBackendNames(globalSearchManager.backends))
	})
}

func buildBackends(cfg *config.Config) []SearchBackend {
	var backends []SearchBackend

	backend := strings.ToLower(cfg.WebSearchBackend)
	apiKey := cfg.WebSearchAPIKey

	switch backend {
	case "tavily":
		if apiKey != "" {
			backends = append(backends, &TavilyBackend{apiKey: apiKey})
			log.Printf("Web search backend: Tavily (API key configured)")
		} else {
			log.Printf("Error: Tavily backend selected but no API key provided")
		}
	case "serper":
		if apiKey != "" {
			backends = append(backends, &SerperBackend{apiKey: apiKey})
			log.Printf("Web search backend: Serper (API key configured)")
		} else {
			log.Printf("Error: Serper backend selected but no API key provided")
		}
	case "brave":
		if apiKey != "" {
			backends = append(backends, &BraveBackend{apiKey: apiKey})
			log.Printf("Web search backend: Brave (API key configured)")
		} else {
			log.Printf("Error: Brave backend selected but no API key provided")
		}
	case "duckduckgo", "":
		backends = append(backends, &DuckDuckGoBackend{})
		log.Printf("Web search backend: DuckDuckGo (free, rate-limited)")
	default:
		log.Printf("Error: Unknown backend '%s', no search backend configured", backend)
	}

	return backends
}

func getBackendNames(backends []SearchBackend) []string {
	names := make([]string, len(backends))
	for i, b := range backends {
		names[i] = b.Name()
	}
	return names
}

// Search performs a search using configured backend (no fallback)
func (m *SearchManager) Search(query string, numResults int) ([]SearchResult, string, error) {
	m.mu.RLock()
	backends := m.backends
	m.mu.RUnlock()

	if len(backends) == 0 {
		return nil, "", fmt.Errorf("no search backend configured")
	}

	// Use the configured backend (first one, no fallback)
	backend := backends[0]
	results, err := backend.Search(query, numResults)
	if err != nil {
		return nil, backend.Name(), err
	}
	return results, backend.Name(), nil
}

// ========================
// Tavily Backend
// ========================

type TavilyBackend struct {
	apiKey string
}

func (t *TavilyBackend) Name() string { return "tavily" }

type tavilyRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results"`
}

type tavilyResponse struct {
	Results []struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Content string `json:"content"`
	} `json:"results"`
}

func (t *TavilyBackend) Search(query string, numResults int) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	reqBody := tavilyRequest{
		Query:      query,
		MaxResults: numResults,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.tavily.com/search", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+t.apiKey)

	client := getHTTPClient()
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	var tavilyResp tavilyResponse
	if err := json.NewDecoder(resp.Body).Decode(&tavilyResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	results := make([]SearchResult, 0, len(tavilyResp.Results))
	for _, r := range tavilyResp.Results {
		if !isValidURL(r.URL) {
			continue
		}
		results = append(results, SearchResult{
			Title:   r.Title,
			URL:     r.URL,
			Snippet: r.Content,
			Domain:  extractDomain(r.URL),
		})
	}

	return results, nil
}

// ========================
// Serper Backend
// ========================

type SerperBackend struct {
	apiKey string
}

func (s *SerperBackend) Name() string { return "serper" }

type serperRequest struct {
	Q   string `json:"q"`
	Num int    `json:"num"`
}

type serperResponse struct {
	Organic []struct {
		Title   string `json:"title"`
		Link    string `json:"link"`
		Snippet string `json:"snippet"`
	} `json:"organic"`
}

func (s *SerperBackend) Search(query string, numResults int) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	reqBody := serperRequest{
		Q:   query,
		Num: numResults,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", "https://google.serper.dev/search", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-KEY", s.apiKey)

	client := getHTTPClient()
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	var serperResp serperResponse
	if err := json.NewDecoder(resp.Body).Decode(&serperResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	results := make([]SearchResult, 0, len(serperResp.Organic))
	for _, r := range serperResp.Organic {
		if !isValidURL(r.Link) {
			continue
		}
		results = append(results, SearchResult{
			Title:   r.Title,
			URL:     r.Link,
			Snippet: r.Snippet,
			Domain:  extractDomain(r.Link),
		})
	}

	return results, nil
}

// ========================
// Brave Backend
// ========================

type BraveBackend struct {
	apiKey string
}

func (b *BraveBackend) Name() string { return "brave" }

type braveResponse struct {
	Web struct {
		Results []struct {
			Title       string `json:"title"`
			URL         string `json:"url"`
			Description string `json:"description"`
		} `json:"results"`
	} `json:"web"`
}

func (b *BraveBackend) Search(query string, numResults int) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	searchURL := fmt.Sprintf("https://api.search.brave.com/res/v1/web/search?q=%s&count=%d",
		url.QueryEscape(query), numResults)

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Subscription-Token", b.apiKey)

	client := getHTTPClient()
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: status %d, body: %s", resp.StatusCode, string(body))
	}

	var braveResp braveResponse
	if err := json.NewDecoder(resp.Body).Decode(&braveResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	results := make([]SearchResult, 0, len(braveResp.Web.Results))
	for _, r := range braveResp.Web.Results {
		if !isValidURL(r.URL) {
			continue
		}
		results = append(results, SearchResult{
			Title:   r.Title,
			URL:     r.URL,
			Snippet: r.Description,
			Domain:  extractDomain(r.URL),
		})
	}

	return results, nil
}

// ========================
// DuckDuckGo Backend (fallback)
// ========================

type DuckDuckGoBackend struct{}

func (d *DuckDuckGoBackend) Name() string { return "duckduckgo" }

func (d *DuckDuckGoBackend) Search(query string, numResults int) ([]SearchResult, error) {
	return searchDuckDuckGo(query, numResults)
}

// ========================
// URL validation (XSS prevention)
// ========================

// isValidURL validates URL to prevent XSS attacks
// Only allows http and https schemes, rejects javascript:, data:, vbscript:, etc.
func isValidURL(urlStr string) bool {
	if urlStr == "" {
		return false
	}

	parsed, err := url.Parse(urlStr)
	if err != nil {
		return false
	}

	// Only allow safe schemes
	scheme := strings.ToLower(parsed.Scheme)
	if scheme != "http" && scheme != "https" {
		return false
	}

	// Must have a valid host
	if parsed.Host == "" {
		return false
	}

	return true
}

// extractDomain extracts the domain from a URL
func extractDomain(urlStr string) string {
	parsed, err := url.Parse(urlStr)
	if err != nil {
		return ""
	}
	return parsed.Host
}

// ========================
// Search with retry mechanism
// ========================

// searchDuckDuckGo performs a search using DuckDuckGo's HTML interface with retry
func searchDuckDuckGo(query string, numResults int) ([]SearchResult, error) {
	if numResults <= 0 {
		numResults = defaultNumResults
	}
	if numResults > maxResultsLimit {
		numResults = maxResultsLimit
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		results, err := doSearchRequest(query, numResults)
		if err == nil {
			return results, nil
		}

		lastErr = err

		// Don't retry on non-transient errors
		if strings.Contains(err.Error(), "status 4") {
			break
		}

		// Exponential backoff
		if attempt < maxRetries-1 {
			delay := retryBaseDelay * time.Duration(1<<attempt)
			time.Sleep(delay)
			log.Printf("Retrying search (attempt %d/%d) after error: %v", attempt+2, maxRetries, err)
		}
	}

	return nil, lastErr
}

// doSearchRequest performs a single search request with concurrency control
func doSearchRequest(query string, numResults int) ([]SearchResult, error) {
	// Acquire semaphore to limit concurrent outgoing requests
	select {
	case concurrentSem <- struct{}{}:
		defer func() { <-concurrentSem }()
	default:
		// If semaphore is full, return rate limit error immediately
		return nil, fmt.Errorf("%s: too many concurrent requests", ErrCodeRateLimited)
	}

	// Add random delay to avoid detection (1-3 seconds)
	randomDelay()

	// Use DuckDuckGo HTML interface
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", url.QueryEscape(query))

	client := getHTTPClient()

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers to mimic a browser request with rotating User-Agent
	req.Header.Set("User-Agent", getRandomUserAgent())
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7")
	req.Header.Set("Accept-Encoding", "gzip, deflate, br")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("%s: %w", ErrCodeTimeout, err)
		}
		return nil, fmt.Errorf("%s: %w", ErrCodeSearchFailed, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: status %d", ErrCodeUpstreamError, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return parseDuckDuckGoHTML(string(body), numResults)
}

// ========================
// HTML parsing (using pre-compiled regex)
// ========================

// parseDuckDuckGoHTML parses DuckDuckGo HTML search results
func parseDuckDuckGoHTML(htmlContent string, maxResults int) ([]SearchResult, error) {
	var results []SearchResult

	// Match result links - the href contains a redirect URL
	linkMatches := linkPattern.FindAllStringSubmatch(htmlContent, -1)

	// Match snippets
	snippetMatches := snippetPattern.FindAllStringSubmatch(htmlContent, -1)

	// Alternative pattern for snippets (sometimes in span)
	if len(snippetMatches) == 0 {
		snippetMatches = snippetAltPattern.FindAllStringSubmatch(htmlContent, -1)
	}

	for i := 0; i < len(linkMatches) && i < maxResults; i++ {
		if len(linkMatches[i]) < 3 {
			continue
		}

		rawURL := linkMatches[i][1]
		title := strings.TrimSpace(linkMatches[i][2])

		// Clean up title - decode HTML entities using standard library
		title = html.UnescapeString(title)
		title = cleanExtraWhitespace(title)

		// Extract actual URL from DuckDuckGo redirect
		actualURL := extractActualURL(rawURL)
		if actualURL == "" {
			continue
		}

		// Security: Validate URL to prevent XSS
		if !isValidURL(actualURL) {
			log.Printf("Skipping invalid URL: %s", actualURL)
			continue
		}

		// Get snippet if available
		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = cleanHTMLTags(snippetMatches[i][1])
			snippet = html.UnescapeString(snippet)
			snippet = cleanExtraWhitespace(snippet)
		}

		// Skip if URL is DuckDuckGo internal
		if strings.Contains(actualURL, "duckduckgo.com") {
			continue
		}

		results = append(results, SearchResult{
			Title:   title,
			URL:     actualURL,
			Snippet: snippet,
			Domain:  extractDomain(actualURL),
		})
	}

	// If regex parsing failed, try a simpler approach
	if len(results) == 0 {
		results = parseSimpleHTML(htmlContent, maxResults)
	}

	return results, nil
}

// parseSimpleHTML is a fallback parser for DuckDuckGo results
func parseSimpleHTML(htmlContent string, maxResults int) []SearchResult {
	var results []SearchResult

	// Split by result blocks
	blocks := strings.Split(htmlContent, "result__body")

	for i, block := range blocks {
		if i == 0 || i > maxResults {
			continue
		}

		// Extract URL using pre-compiled pattern
		urlMatch := uddgPattern.FindStringSubmatch(block)
		if len(urlMatch) < 2 {
			continue
		}
		actualURL, _ := url.QueryUnescape(urlMatch[1])
		if actualURL == "" || strings.Contains(actualURL, "duckduckgo") {
			continue
		}

		// Security: Validate URL
		if !isValidURL(actualURL) {
			continue
		}

		// Extract title using pre-compiled pattern
		titleMatch := titlePattern.FindStringSubmatch(block)
		title := ""
		if len(titleMatch) > 1 {
			title = html.UnescapeString(strings.TrimSpace(titleMatch[1]))
		}

		// Extract snippet using pre-compiled pattern
		snippetMatch := snippetFallbackPattern.FindStringSubmatch(block)
		snippet := ""
		if len(snippetMatch) > 1 {
			snippet = html.UnescapeString(strings.TrimSpace(snippetMatch[1]))
		}

		if title != "" {
			results = append(results, SearchResult{
				Title:   title,
				URL:     actualURL,
				Snippet: snippet,
				Domain:  extractDomain(actualURL),
			})
		}
	}

	return results
}

// extractActualURL extracts the actual URL from DuckDuckGo's redirect URL
func extractActualURL(ddgURL string) string {
	// DuckDuckGo uses a redirect format like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
	if strings.Contains(ddgURL, "uddg=") {
		parsed, err := url.Parse(ddgURL)
		if err == nil {
			uddg := parsed.Query().Get("uddg")
			if uddg != "" {
				return uddg
			}
		}
		// Try regex extraction as fallback using pre-compiled pattern
		matches := uddgPattern.FindStringSubmatch(ddgURL)
		if len(matches) > 1 {
			decoded, err := url.QueryUnescape(matches[1])
			if err == nil {
				return decoded
			}
			return matches[1]
		}
	}

	// If it's already a direct URL
	if strings.HasPrefix(ddgURL, "http://") || strings.HasPrefix(ddgURL, "https://") {
		return ddgURL
	}

	return ""
}

// ========================
// Utility functions (using pre-compiled regex)
// ========================

// cleanHTMLTags removes HTML tags from a string
func cleanHTMLTags(s string) string {
	return strings.TrimSpace(htmlTagPattern.ReplaceAllString(s, " "))
}

// cleanExtraWhitespace normalizes whitespace in a string
func cleanExtraWhitespace(s string) string {
	return strings.TrimSpace(whitespacePattern.ReplaceAllString(s, " "))
}

// ========================
// HTTP Handler
// ========================

// getClientIP extracts client IP from request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxied requests)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	// Fall back to RemoteAddr
	ip := r.RemoteAddr
	if idx := strings.LastIndex(ip, ":"); idx != -1 {
		ip = ip[:idx]
	}
	return ip
}

// sendErrorResponse sends a JSON error response with code
func sendErrorResponse(w http.ResponseWriter, query string, code SearchErrorCode, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(WebSearchResponse{
		Query: query,
		Error: errorMessages[code],
		Code:  code,
	})
}

// WebSearchHandler handles web search requests
func WebSearchHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		// Handle preflight
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Only allow POST requests
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Rate limiting check (per-IP + global) - only for DuckDuckGo fallback
		clientIP := getClientIP(r)

		// Parse request body
		var req WebSearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("Error parsing search request: %v", err)
			sendErrorResponse(w, "", ErrCodeInvalidRequest, http.StatusBadRequest)
			return
		}

		// Validate query
		if req.Query == "" {
			sendErrorResponse(w, req.Query, ErrCodeEmptyQuery, http.StatusBadRequest)
			return
		}
		if len(req.Query) > maxQueryLength {
			sendErrorResponse(w, req.Query, ErrCodeQueryTooLong, http.StatusBadRequest)
			return
		}

		log.Printf("Web search request: query=%q, num_results=%d, ip=%s", req.Query, req.NumResults, clientIP)

		// Check if search manager is initialized
		if globalSearchManager == nil {
			log.Printf("Search manager not initialized, using DuckDuckGo directly")
			// Rate limit for DuckDuckGo
			allowed, errCode := globalRateLimiter.isAllowed(clientIP)
			if !allowed {
				log.Printf("Rate limit exceeded for IP: %s (code: %s)", clientIP, errCode)
				sendErrorResponse(w, "", errCode, http.StatusTooManyRequests)
				return
			}
			results, err := searchDuckDuckGo(req.Query, req.NumResults)
			if err != nil {
				log.Printf("Search error: %v", err)
				sendErrorResponse(w, req.Query, ErrCodeSearchFailed, http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(WebSearchResponse{
				Query:   req.Query,
				Results: results,
				Backend: "duckduckgo",
			})
			return
		}

		// Check if no backend configured
		globalSearchManager.mu.RLock()
		numBackends := len(globalSearchManager.backends)
		globalSearchManager.mu.RUnlock()

		if numBackends == 0 {
			log.Printf("No search backend configured")
			sendErrorResponse(w, req.Query, ErrCodeSearchFailed, http.StatusServiceUnavailable)
			return
		}

		// Apply rate limit for DuckDuckGo backend
		globalSearchManager.mu.RLock()
		backendName := globalSearchManager.backends[0].Name()
		globalSearchManager.mu.RUnlock()

		if backendName == "duckduckgo" {
			allowed, errCode := globalRateLimiter.isAllowed(clientIP)
			if !allowed {
				log.Printf("Rate limit exceeded for IP: %s (code: %s)", clientIP, errCode)
				sendErrorResponse(w, "", errCode, http.StatusTooManyRequests)
				return
			}
		}

		// Use search manager
		results, backend, err := globalSearchManager.Search(req.Query, req.NumResults)
		if err != nil {
			log.Printf("Search error: %v", err)

			// Determine appropriate error code
			errCode := ErrCodeSearchFailed
			statusCode := http.StatusInternalServerError

			errStr := err.Error()
			if strings.Contains(errStr, string(ErrCodeTimeout)) {
				errCode = ErrCodeTimeout
				statusCode = http.StatusGatewayTimeout
			} else if strings.Contains(errStr, string(ErrCodeUpstreamError)) {
				errCode = ErrCodeUpstreamError
				statusCode = http.StatusBadGateway
			}

			sendErrorResponse(w, req.Query, errCode, statusCode)
			return
		}

		log.Printf("Web search completed: query=%q, results=%d, backend=%s", req.Query, len(results), backend)

		// Send response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(WebSearchResponse{
			Query:   req.Query,
			Results: results,
			Backend: backend,
		})
	}
}
