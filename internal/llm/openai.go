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

// openaiRequest is the OpenAI Chat Completions API request body.
type openaiRequest struct {
	Model       string          `json:"model"`
	MaxTokens   int             `json:"max_tokens"`
	Messages    []openaiMessage `json:"messages"`
	Temperature *float64        `json:"temperature,omitempty"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// openaiResponse is the OpenAI Chat Completions API response body.
type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// openaiErrorResponse represents an OpenAI API error.
type openaiErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// OpenAIClient implements Client for OpenAI-compatible Chat Completions APIs.
// Works with OpenAI, Azure OpenAI, and any OpenAI-compatible provider.
type OpenAIClient struct {
	apiKey     string
	baseURL    string
	model      string
	http       *http.Client
	maxRetries int
}

// NewOpenAIClient creates a new OpenAI-compatible API client.
func NewOpenAIClient(apiKey, baseURL, model string) *OpenAIClient {
	return &OpenAIClient{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		http: &http.Client{
			Timeout: 120 * time.Second,
		},
		maxRetries: 3,
	}
}

// Complete sends a message to the OpenAI-compatible API and returns the response.
func (c *OpenAIClient) Complete(ctx context.Context, req Request) (*Response, error) {
	model := req.Model
	if model == "" {
		model = c.model
	}

	// Convert to OpenAI message format.
	// OpenAI uses a system message in the messages array.
	var messages []openaiMessage
	if req.System != "" {
		messages = append(messages, openaiMessage{
			Role:    "system",
			Content: req.System,
		})
	}
	for _, m := range req.Messages {
		messages = append(messages, openaiMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	apiReq := openaiRequest{
		Model:       model,
		MaxTokens:   req.MaxTokens,
		Messages:    messages,
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

		httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("creating request: %w", err)
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

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
			var apiResp openaiResponse
			if err := json.Unmarshal(respBody, &apiResp); err != nil {
				return nil, fmt.Errorf("unmarshalling response: %w", err)
			}
			return convertOpenAIResponse(&apiResp), nil
		}

		var errResp openaiErrorResponse
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

func convertOpenAIResponse(apiResp *openaiResponse) *Response {
	var text string
	if len(apiResp.Choices) > 0 {
		text = apiResp.Choices[0].Message.Content
	}
	return &Response{
		Text:         text,
		InputTokens:  apiResp.Usage.PromptTokens,
		OutputTokens: apiResp.Usage.CompletionTokens,
	}
}
