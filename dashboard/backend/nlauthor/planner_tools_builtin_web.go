package nlauthor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	PlannerToolSourceBuiltinWeb = "builtin_web"
)

func NewBuiltinWebToolSource() PlannerToolSource {
	return builtinWebToolSource{
		client: &http.Client{Timeout: 15 * time.Second},
	}
}

type builtinWebToolSource struct {
	client *http.Client
}

func (s builtinWebToolSource) SourceName() string {
	return PlannerToolSourceBuiltinWeb
}

func (s builtinWebToolSource) Tools(_ Session, _ TurnRequest) []PlannerTool {
	return []PlannerTool{
		staticPlannerTool{
			definition: PlannerToolDefinition{
				Name:        "fetch_raw_url",
				Description: "Fetch raw text content from an HTTP or HTTPS URL.",
				InputSchema: objectSchema(requiredStringField("url")),
				Readonly:    true,
				Source:      PlannerToolSourceBuiltinWeb,
			},
			invokeFn: func(ctx context.Context, _ Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				var payload struct {
					URL string `json:"url"`
				}
				if err := json.Unmarshal(arguments, &payload); err != nil {
					return PlannerToolResult{}, fmt.Errorf("invalid fetch_raw_url arguments: %w", err)
				}
				targetURL := strings.TrimSpace(payload.URL)
				if targetURL == "" {
					return PlannerToolResult{}, fmt.Errorf("url is required")
				}
				parsed, err := url.Parse(targetURL)
				if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
					return PlannerToolResult{}, fmt.Errorf("invalid url, must use http or https")
				}

				request, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
				if err != nil {
					return PlannerToolResult{}, fmt.Errorf("build fetch request: %w", err)
				}
				request.Header.Set("Accept", "text/plain, application/yaml, application/json, */*")
				request.Header.Set("User-Agent", "semantic-router-nl-planner/1.0")

				response, err := s.client.Do(request)
				if err != nil {
					return PlannerToolResult{}, fmt.Errorf("execute fetch request: %w", err)
				}
				defer response.Body.Close()
				if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
					return PlannerToolResult{}, fmt.Errorf("remote returned %d", response.StatusCode)
				}
				body, err := io.ReadAll(io.LimitReader(response.Body, 2*1024*1024))
				if err != nil {
					return PlannerToolResult{}, fmt.Errorf("read fetch response: %w", err)
				}
				return marshalToolJSON(map[string]any{
					"url":     targetURL,
					"content": string(body),
				})
			},
		},
	}
}
