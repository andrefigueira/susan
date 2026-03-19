// Package llm defines the provider-agnostic LLM client interface.
//
// This abstraction enables cross-model evaluation (addressing the shared
// training bias concern when Claude evaluates Claude) and multi-model
// experiments (testing ISC predictions on non-Anthropic models).
package llm

import "context"

// Client is the interface that all LLM providers must implement.
type Client interface {
	Complete(ctx context.Context, req Request) (*Response, error)
}

// Request is the provider-agnostic API request.
// Temperature uses a pointer to distinguish "not set" from "set to 0.0".
type Request struct {
	Model       string    `json:"model,omitempty"`
	MaxTokens   int       `json:"max_tokens"`
	System      string    `json:"system,omitempty"`
	Messages    []Message `json:"messages"`
	Temperature *float64  `json:"temperature,omitempty"`
}

// Message represents a conversation message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Response is the provider-agnostic API response.
type Response struct {
	Text         string
	InputTokens  int
	OutputTokens int
}

// NewTemperature returns a pointer to a float64, for use in Request.Temperature.
func NewTemperature(t float64) *float64 {
	return &t
}
