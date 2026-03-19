package llm

import "time"

// isRetryable returns true for HTTP status codes that warrant a retry.
func isRetryable(statusCode int) bool {
	return statusCode == 429 || statusCode >= 500
}

// retryDelay returns the backoff duration for a given retry attempt.
// Uses exponential backoff: 2s, 4s, 8s.
func retryDelay(attempt int) time.Duration {
	return time.Duration(1<<uint(attempt)) * time.Second
}
