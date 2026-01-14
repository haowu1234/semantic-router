package builtin

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// WebSearch is a built-in tool for web searching
type WebSearch struct {
	client *http.Client
}

// NewWebSearch creates a new WebSearch tool
func NewWebSearch() *WebSearch {
	return &WebSearch{
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Name returns the tool name (matches tools_db.json)
func (w *WebSearch) Name() string {
	return "search_web"
}

// Description returns the tool description
func (w *WebSearch) Description() string {
	return "Search the web for current information. Use this for questions about " +
		"recent events, real-time data, or information that may have changed since training."
}

// Parameters returns the tool parameters
func (w *WebSearch) Parameters() []models.ToolParameter {
	return []models.ToolParameter{
		{
			Name:        "query",
			Type:        "string",
			Description: "The search query to look up",
			Required:    true,
		},
		{
			Name:        "max_results",
			Type:        "integer",
			Description: "Maximum number of results to return (1-10)",
			Required:    false,
			Default:     5,
		},
	}
}

// SearchResult represents a single search result
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// Execute performs a web search
func (w *WebSearch) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return nil, fmt.Errorf("query must be a non-empty string")
	}

	maxResults := 5
	if mr, ok := args["max_results"].(float64); ok {
		maxResults = int(mr)
		if maxResults < 1 {
			maxResults = 1
		} else if maxResults > 10 {
			maxResults = 10
		}
	}

	// Check for SerpAPI key
	serpAPIKey := os.Getenv("SERPAPI_KEY")
	if serpAPIKey != "" {
		return w.searchWithSerpAPI(ctx, query, maxResults, serpAPIKey)
	}

	// Check for Serper API key
	serperAPIKey := os.Getenv("SERPER_API_KEY")
	if serperAPIKey != "" {
		return w.searchWithSerper(ctx, query, maxResults, serperAPIKey)
	}

	// Return mock results if no API key is configured
	return w.mockSearch(query, maxResults)
}

// searchWithSerpAPI uses SerpAPI for real search results
func (w *WebSearch) searchWithSerpAPI(ctx context.Context, query string, maxResults int, apiKey string) (interface{}, error) {
	baseURL := "https://serpapi.com/search"
	params := url.Values{}
	params.Set("q", query)
	params.Set("api_key", apiKey)
	params.Set("num", fmt.Sprintf("%d", maxResults))

	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"?"+params.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search API error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	// Extract organic results
	results := []SearchResult{}
	if organic, ok := result["organic_results"].([]interface{}); ok {
		for i, item := range organic {
			if i >= maxResults {
				break
			}
			if itemMap, ok := item.(map[string]interface{}); ok {
				sr := SearchResult{}
				if title, ok := itemMap["title"].(string); ok {
					sr.Title = title
				}
				if link, ok := itemMap["link"].(string); ok {
					sr.URL = link
				}
				if snippet, ok := itemMap["snippet"].(string); ok {
					sr.Snippet = snippet
				}
				results = append(results, sr)
			}
		}
	}

	return map[string]interface{}{
		"query":   query,
		"results": results,
		"source":  "serpapi",
	}, nil
}

// searchWithSerper uses Serper API for real search results
func (w *WebSearch) searchWithSerper(ctx context.Context, query string, maxResults int, apiKey string) (interface{}, error) {
	baseURL := "https://google.serper.dev/search"

	payload := map[string]interface{}{
		"q":   query,
		"num": maxResults,
	}
	payloadBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	req.Header.Set("X-API-KEY", apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := w.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search API error: %s", string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %v", err)
	}

	// Extract organic results
	results := []SearchResult{}
	if organic, ok := result["organic"].([]interface{}); ok {
		for i, item := range organic {
			if i >= maxResults {
				break
			}
			if itemMap, ok := item.(map[string]interface{}); ok {
				sr := SearchResult{}
				if title, ok := itemMap["title"].(string); ok {
					sr.Title = title
				}
				if link, ok := itemMap["link"].(string); ok {
					sr.URL = link
				}
				if snippet, ok := itemMap["snippet"].(string); ok {
					sr.Snippet = snippet
				}
				results = append(results, sr)
			}
		}
	}

	return map[string]interface{}{
		"query":   query,
		"results": results,
		"source":  "serper",
	}, nil
}

// mockSearch returns mock search results when no API key is configured
func (w *WebSearch) mockSearch(query string, maxResults int) (interface{}, error) {
	// Generate mock results based on query
	results := []SearchResult{
		{
			Title:   fmt.Sprintf("Search results for: %s - Wikipedia", query),
			URL:     fmt.Sprintf("https://en.wikipedia.org/wiki/%s", url.QueryEscape(query)),
			Snippet: fmt.Sprintf("Wikipedia article about %s. This is a mock result. Configure SERPAPI_KEY or SERPER_API_KEY for real search results.", query),
		},
		{
			Title:   fmt.Sprintf("%s - Latest News and Updates", query),
			URL:     "https://news.example.com/article",
			Snippet: "This is a mock news result. To enable real web search, set the SERPAPI_KEY or SERPER_API_KEY environment variable.",
		},
		{
			Title:   fmt.Sprintf("Understanding %s: A Complete Guide", query),
			URL:     "https://guide.example.com/complete-guide",
			Snippet: "A comprehensive guide covering all aspects. Note: This is mock data for demonstration purposes.",
		},
	}

	if len(results) > maxResults {
		results = results[:maxResults]
	}

	return map[string]interface{}{
		"query":   query,
		"results": results,
		"source":  "mock",
		"note":    "Configure SERPAPI_KEY or SERPER_API_KEY environment variable for real search results",
	}, nil
}
