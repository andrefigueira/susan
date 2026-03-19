package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"
)

const anthropicAPIVersion = "2023-06-01"

// anthropicRequest is the Anthropic Messages API request body.
type anthropicRequest struct {
	Model       string    `json:"model"`
	MaxTokens   int       `json:"max_tokens"`
	System      string    `json:"system,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
}

// anthropicResponse is the Anthropic Messages API response body.
type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// anthropicErrorResponse represents an Anthropic API error.
type anthropicErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// AnthropicClient implements Client for the Anthropic Messages API.
type AnthropicClient struct {
	apiKey     string
	baseURL    string
	model      string
	http       *http.Client
	maxRetries int
}

// NewAnthropicClient creates a new Anthropic API client.
func NewAnthropicClient(apiKey, baseURL, model string) *AnthropicClient {
	return &AnthropicClient{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		http: &http.Client{
			Timeout: 120 * time.Second,
		},
		maxRetries: 3,
	}
}

// Complete sends a message to the Anthropic API and returns the response.
func (c *AnthropicClient) Complete(ctx context.Context, req Request) (*Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}

	apiReq := anthropicRequest{
		Model:       model,
		MaxTokens:   req.MaxTokens,
		System:      req.System,
		Messages:    req.Messages,
		Temperature: req.Temperature,
	}

	body, err := json.Marshal(apiReq)
	if err != nil {
		return nil, fmt.Errorf("marshalling request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			backoff := retryDelay(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/messages", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("x-api-key", c.apiKey)
		httpReq.Header.Set("anthropic-version", anthropicAPIVersion)

		resp, err := c.http.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("sending request: %w", err)
			if attempt < c.maxRetries {
				continue
			}
			return nil, lastErr
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("reading response: %w", err)
			if attempt < c.maxRetries {
				continue
			}
			return nil, lastErr
		}

		if resp.StatusCode == http.StatusOK {
			var apiResp anthropicResponse
			if err := json.Unmarshal(respBody, &apiResp); err != nil {
				return nil, fmt.Errorf("unmarshalling response: %w", err)
			}
			return convertAnthropicResponse(&apiResp), nil
		}

		var errResp anthropicErrorResponse
		if jsonErr := json.Unmarshal(respBody, &errResp); jsonErr == nil && errResp.Error.Message != "" {
			lastErr = fmt.Errorf("API error %d: %s: %s", resp.StatusCode, errResp.Error.Type, errResp.Error.Message)
		} else {
			lastErr = fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
		}

		if isRetryable(resp.StatusCode) && attempt < c.maxRetries {
			if ra := resp.Header.Get("Retry-After"); ra != "" {
				if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
					select {
					case <-ctx.Done():
						return nil, ctx.Err()
					case <-time.After(time.Duration(secs) * time.Second):
					}
				}
			}
			continue
		}

		return nil, lastErr
	}

	return nil, lastErr
}

func convertAnthropicResponse(apiResp *anthropicResponse) *Response {
	var text string
	for _, c := range apiResp.Content {
		if c.Type == "text" {
			text += c.Text
		}
	}
	return &Response{
		Text:         text,
		InputTokens:  apiResp.Usage.InputTokens,
		OutputTokens: apiResp.Usage.OutputTokens,
	}
}
