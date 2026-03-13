// Package claude provides a minimal Claude API client.
//
// This deliberately avoids the official SDK to keep dependencies minimal
// and to have full control over request construction, which is necessary
// for the disruption mechanisms (token limits, temperature, etc).
package claude

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

const apiVersion = "2023-06-01"

// Message represents a conversation message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request is the API request body.
// Temperature uses a pointer to distinguish between "not set" and "set to 0.0".
type Request struct {
	Model       string    `json:"model"`
	MaxTokens   int       `json:"max_tokens"`
	System      string    `json:"system,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
}

// NewTemperature returns a pointer to a float64, for use in Request.Temperature.
func NewTemperature(t float64) *float64 {
	return &t
}

// Response is the API response body (simplified).
type Response struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Model        string `json:"model"`
	StopReason   string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// Text returns the concatenated text content of the response.
// Returns empty string if no text content blocks exist.
func (r *Response) Text() string {
	var text string
	for _, c := range r.Content {
		if c.Type == "text" {
			text += c.Text
		}
	}
	return text
}

// ErrorResponse represents an API error.
type ErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Client is a minimal Claude API client.
type Client struct {
	apiKey     string
	baseURL    string
	model      string
	http       *http.Client
	maxRetries int
}

// NewClient creates a new Claude API client.
func NewClient(apiKey, baseURL, model string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		http: &http.Client{
			Timeout: 120 * time.Second,
		},
		maxRetries: 3,
	}
}

// Complete sends a message to Claude and returns the response.
// Retries on transient errors (429 rate limit, 5xx server errors)
// with exponential backoff.
func (c *Client) Complete(ctx context.Context, req Request) (*Response, error) {
	if req.Model == "" {
		req.Model = c.model
	}

	body, err := json.Marshal(req)
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
		httpReq.Header.Set("anthropic-version", apiVersion)

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
			var apiResp Response
			if err := json.Unmarshal(respBody, &apiResp); err != nil {
				return nil, fmt.Errorf("unmarshalling response: %w", err)
			}
			return &apiResp, nil
		}

		// Parse error for the message.
		var errResp ErrorResponse
		if jsonErr := json.Unmarshal(respBody, &errResp); jsonErr == nil && errResp.Error.Message != "" {
			lastErr = fmt.Errorf("API error %d: %s: %s", resp.StatusCode, errResp.Error.Type, errResp.Error.Message)
		} else {
			lastErr = fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
		}

		// Retry on 429 (rate limit) and 5xx (server errors).
		if isRetryable(resp.StatusCode) && attempt < c.maxRetries {
			// Respect Retry-After header if present.
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

// isRetryable returns true for HTTP status codes that warrant a retry.
func isRetryable(statusCode int) bool {
	return statusCode == 429 || statusCode >= 500
}

// retryDelay returns the backoff duration for a given retry attempt.
// Uses exponential backoff: 2s, 4s, 8s.
func retryDelay(attempt int) time.Duration {
	return time.Duration(1<<uint(attempt)) * time.Second
}
