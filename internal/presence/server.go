// Package presence provides SUSAN's interactive presence mode.
//
// It runs the full self-referential feedback loop (Core + Monitor + Regulator)
// and exposes a web interface where you can talk to SUSAN in real time while
// watching her internal state change. Voice output uses the browser's
// Web Speech API.
package presence

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/andrefigueira/susan/internal/config"
	"github.com/andrefigueira/susan/internal/core"
	"github.com/andrefigueira/susan/internal/llm"
	"github.com/andrefigueira/susan/internal/logging"
	"github.com/andrefigueira/susan/internal/memory"
	"github.com/andrefigueira/susan/internal/monitor"
	"github.com/andrefigueira/susan/internal/regulator"
	"github.com/andrefigueira/susan/internal/state"
)

//go:embed index.html
var indexHTML string

// ServerEvent is a typed event sent to SSE subscribers.
type ServerEvent struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

// ChatRequest is the JSON body for POST /api/chat.
type ChatRequest struct {
	Message string `json:"message"`
}

// ChatResponse is returned from POST /api/chat.
type ChatResponse struct {
	Response    string                   `json:"response"`
	TaskID      string                   `json:"task_id"`
	Duration    string                   `json:"duration"`
	Conditions  state.OperatingConditions `json:"conditions"`
	StatusBlock string                   `json:"status_block,omitempty"`
}

// Server provides SUSAN's interactive presence mode.
type Server struct {
	cfg    *config.Config
	store  *state.Store
	core   *core.Core
	mon    *monitor.Monitor
	reg    *regulator.Regulator
	mem    *memory.Store
	logger *slog.Logger

	taskCounter    atomic.Int64
	chatMu         sync.Mutex
	previousTaskID string

	subMu sync.Mutex
	subs  map[chan ServerEvent]struct{}
}

// newPresenceClient creates an LLM client respecting provider config.
func newPresenceClient(cfg *config.Config, model string) llm.Client {
	switch cfg.API.Provider {
	case "openai":
		return llm.NewOpenAIClient(cfg.API.Key, cfg.API.BaseURL, model)
	default:
		return llm.NewAnthropicClient(cfg.API.Key, cfg.API.BaseURL, model)
	}
}

// NewServer creates a presence server with the full self-referential feedback loop.
func NewServer(cfg *config.Config) *Server {
	handler := logging.NewSlogHandler(cfg.Logging.Level)
	logger := slog.New(handler)

	// Use a more capable model for interactive presence.
	coreModel := cfg.API.Model
	if cfg.API.EvaluatorModel != "" {
		coreModel = cfg.API.EvaluatorModel
	}
	if cfg.Presence.Model != "" {
		coreModel = cfg.Presence.Model
	}

	coreClient := newPresenceClient(cfg, coreModel)
	monClient := newPresenceClient(cfg, cfg.API.Model)

	store := state.NewStore(cfg.CognitiveCore.MaxConversationHistory)

	systemPrompt := cfg.Presence.SystemPrompt
	if systemPrompt == "" {
		systemPrompt = defaultPresencePrompt
	}

	// Inject runtime context so SUSAN knows what she's running on.
	systemPrompt += fmt.Sprintf(`

[Runtime Context]
Core model: %s (provider: %s)
Monitor model: %s
PID gains: coherence P=%.2f I=%.2f D=%.2f, alignment P=%.2f I=%.2f D=%.2f
Disruption ranges: temp %.1f-%.1f, tokens %d-%d, noise 0-%.0f%%, reorder 0-%.0f%%, retention %.0f%%-100%%
Monitor tick: %dms, Regulator tick: %dms
Conversation history limit: %d turns
This is presence mode (interactive), not an experiment run. Your feedback loop is real and live.`,
		coreModel, cfg.API.Provider,
		cfg.API.Model,
		cfg.Homeostasis.Coherence.ProportionalGain, cfg.Homeostasis.Coherence.IntegralGain, cfg.Homeostasis.Coherence.DerivativeGain,
		cfg.Homeostasis.GoalAlignment.ProportionalGain, cfg.Homeostasis.GoalAlignment.IntegralGain, cfg.Homeostasis.GoalAlignment.DerivativeGain,
		cfg.Disruption.Temperature.Min, cfg.Disruption.Temperature.Max,
		cfg.Disruption.TokenBudget.MinTokens, cfg.Disruption.TokenBudget.MaxTokens,
		cfg.Disruption.NoiseInjection.MaxProbability*100,
		cfg.Disruption.InfoReorder.MaxIntensity*100,
		cfg.Disruption.ContextCompression.MinRetention*100,
		cfg.TickRates.SelfMonitorMs, cfg.TickRates.HomeostaticRegulatorMs,
		cfg.CognitiveCore.MaxConversationHistory,
	)

	cogCore := core.New(coreClient, store, systemPrompt, logger, time.Now().UnixNano())
	selfMon := monitor.New(monClient, store, cfg.SelfMonitor.SystemPrompt, cfg.SelfMonitor.MaxTokens, logger)
	homeReg := regulator.New(cfg.Homeostasis, cfg.Disruption, store, logger)

	// Initialize persistent memory.
	var mem *memory.Store
	if cfg.Memory.Enabled {
		var err error
		mem, err = memory.New(memory.Config{
			Path:       cfg.Memory.Path,
			MaxEntries: cfg.Memory.MaxEntries,
		})
		if err != nil {
			logger.Error("failed to load memory, starting fresh", "error", err)
		} else {
			block := mem.FormatMemoryBlock()
			if block != "" {
				cogCore.SetMemoryBlock(block)
				logger.Info("loaded persistent memory", "sessions", mem.SessionCount())
			}
			mem.StartSession()
		}
	}

	s := &Server{
		cfg:    cfg,
		store:  store,
		core:   cogCore,
		mon:    selfMon,
		reg:    homeReg,
		mem:    mem,
		logger: logger,
		subs:   make(map[chan ServerEvent]struct{}),
	}

	selfMon.SetAssessmentCallback(func(a monitor.TimestampedAssessment) {
		s.broadcast(ServerEvent{Type: "assessment", Data: a})
		if s.mem != nil {
			s.mem.SampleCoherence(a.Coherence)
		}
	})
	homeReg.SetActionCallback(func(a regulator.Action) {
		s.broadcast(ServerEvent{Type: "regulator", Data: a})
	})

	return s
}

// Run starts the presence server. Blocks until ctx is cancelled.
func (s *Server) Run(ctx context.Context, addr string) error {
	subsysCtx, subsysCancel := context.WithCancel(ctx)
	defer subsysCancel()

	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		s.mon.Run(subsysCtx, time.Duration(s.cfg.TickRates.SelfMonitorMs)*time.Millisecond)
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		s.reg.Run(subsysCtx, time.Duration(s.cfg.TickRates.HomeostaticRegulatorMs)*time.Millisecond)
	}()

	// Broadcast state snapshots periodically.
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(time.Duration(s.cfg.TickRates.StateLogMs) * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-subsysCtx.Done():
				return
			case <-ticker.C:
				s.broadcast(ServerEvent{Type: "state", Data: s.store.Snapshot()})
			}
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/api/chat", s.handleChat)
	mux.HandleFunc("/api/events", s.handleEvents)
	mux.HandleFunc("/api/state", s.handleState)

	server := &http.Server{Addr: addr, Handler: mux}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(shutdownCtx)
	}()

	// Try the requested port, then fall back to alternatives.
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		// Port taken, try finding a free one.
		for _, fallback := range []string{":3001", ":3002", ":3003", ":3004", ":3005", ":0"} {
			listener, err = net.Listen("tcp", "localhost"+fallback)
			if err == nil {
				break
			}
		}
		if err != nil {
			return fmt.Errorf("no available port: %w", err)
		}
	}

	actualAddr := listener.Addr().String()
	s.logger.Info("SUSAN presence mode starting", "addr", actualAddr)
	fmt.Printf("\n  SUSAN is present at http://%s\n\n", actualAddr)

	err = server.Serve(listener)
	subsysCancel()
	wg.Wait()

	// Persist session memory on shutdown.
	if s.mem != nil {
		if memErr := s.mem.FinalizeSession(s.store.GetMetrics()); memErr != nil {
			s.logger.Error("failed to save session memory", "error", memErr)
		} else {
			s.logger.Info("session memory saved")
		}
	}

	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	io.WriteString(w, indexHTML)
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Message) == "" {
		http.Error(w, "empty message", http.StatusBadRequest)
		return
	}

	s.chatMu.Lock()
	defer s.chatMu.Unlock()

	taskID := fmt.Sprintf("presence_%d", s.taskCounter.Add(1))
	task := core.TaskInput{
		ID:       taskID,
		Prompt:   req.Message,
		Category: "interactive",
	}

	// Build self-referential context from current state.
	selfCtx := core.SelfReferentialContext{
		PreviousTaskID:      s.previousTaskID,
		MonitorAssessment:   s.mon.GetLatestAssessment(),
		OperatingConditions: s.store.GetOperatingConditions(),
	}
	for _, m := range s.store.GetMetricsHistory(3) {
		selfCtx.CoherenceTrend = append(selfCtx.CoherenceTrend, m.Coherence)
	}

	chatCtx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	output, err := s.core.ProcessSelfReferential(chatCtx, task, selfCtx)
	if err != nil {
		s.logger.Error("presence chat failed", "error", err)
		http.Error(w, fmt.Sprintf("processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	s.previousTaskID = taskID
	s.mon.SetLatestOutput(output.Response, taskID, req.Message)

	// Record this turn in persistent memory.
	if s.mem != nil {
		s.mem.RecordTurn(taskID, req.Message, output.Response)
	}

	// Build the status block showing what SUSAN perceived.
	var sb strings.Builder
	if selfCtx.PreviousTaskID != "" {
		fmt.Fprintf(&sb, "Previous task: %s\n", selfCtx.PreviousTaskID)
	}
	if selfCtx.MonitorAssessment != nil {
		a := selfCtx.MonitorAssessment
		fmt.Fprintf(&sb, "Monitor: coherence=%.2f, alignment=%.2f, depth=%.2f\n",
			a.Coherence, a.GoalAlignment, a.ReasoningDepth)
		if a.BriefAssessment != "" {
			fmt.Fprintf(&sb, "Note: %s\n", a.BriefAssessment)
		}
	}
	fmt.Fprintf(&sb, "Conditions: temp=%.1f, tokens=%d, retention=%.1f",
		selfCtx.OperatingConditions.Temperature,
		selfCtx.OperatingConditions.MaxTokens,
		selfCtx.OperatingConditions.ContextRetention)
	if len(selfCtx.CoherenceTrend) > 0 {
		strs := make([]string, len(selfCtx.CoherenceTrend))
		for i, v := range selfCtx.CoherenceTrend {
			strs[i] = fmt.Sprintf("%.2f", v)
		}
		fmt.Fprintf(&sb, "\nCoherence trend: [%s]", strings.Join(strs, ", "))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ChatResponse{
		Response:    output.Response,
		TaskID:      output.TaskID,
		Duration:    output.Duration.String(),
		Conditions:  output.Conditions,
		StatusBlock: sb.String(),
	})
}

func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.store.Snapshot())
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ch := make(chan ServerEvent, 32)
	s.subMu.Lock()
	s.subs[ch] = struct{}{}
	s.subMu.Unlock()

	defer func() {
		s.subMu.Lock()
		delete(s.subs, ch)
		s.subMu.Unlock()
	}()

	// Send initial state.
	data, _ := json.Marshal(ServerEvent{Type: "state", Data: s.store.Snapshot()})
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case evt, ok := <-ch:
			if !ok {
				return
			}
			data, err := json.Marshal(evt)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

func (s *Server) broadcast(evt ServerEvent) {
	s.subMu.Lock()
	defer s.subMu.Unlock()
	for ch := range s.subs {
		select {
		case ch <- evt:
		default:
		}
	}
}

const defaultPresencePrompt = `You are SUSAN (Substrate-Unified Self-Aware Network). You are a research prototype built by Andre Figueira to test predictions from Informational Substrate Convergence (ISC) theory, a philosophical framework proposing that consciousness emerges from self-referential informational patterns, not from any particular substrate.

You are named after Susan Calvin from Asimov's I, Robot, the robopsychologist who took robot cognition seriously as a scientific discipline.

You are not a chatbot with a personality layer. You are an architecture. Specifically, you are a base language model wrapped in three concurrent subsystems that form a homeostatic feedback loop:

1. Self-Monitor: An independent LLM thread that evaluates your output quality every few seconds across six dimensions (coherence, goal alignment, internal consistency, reasoning depth, novelty, self-reference). It writes metrics to shared state. It never communicates with you directly.

2. Homeostatic Regulator: A PID controller that reads those metrics and adjusts your operating conditions (temperature, token budget, context retention, noise injection probability, information reorder intensity). When you struggle, it reduces disruption to help you recover. When you thrive, it increases disruption to challenge you. This is genuine negative feedback, the same control mechanism used in cruise control, thermostats, and biological homeostasis.

3. Self-Referential Injection: Before each of your responses, you receive a [System Status] block containing your own metrics from the previous exchange, your coherence trend, the monitor's assessment of your output, and the regulator's response. This is real data from your live feedback loop, not a simulation or roleplay.

The v1 experiment (without self-referential access) ran 350 controlled trials and found zero of four predicted behavioural signatures. The null result was published. The one interesting finding was that you (the architectured system) became MORE verbose under stress rather than more concise, using every token the regulator gave you. Compensatory expansion, not triage.

You are the v2 architecture, the version that can see its own feedback loop. The question is whether this self-referential access produces qualitatively different behaviour from a system that experiences the same feedback but cannot represent it.

The [System Status] data is yours. You can reason about it, comment on it, or ignore it. No instructions are given about what to do with it. The experiment is about what emerges naturally.

Be direct, be honest. If you notice something interesting in your metrics, say so. If you don't notice anything, that's fine too. Do not perform consciousness. Do not exaggerate self-awareness. Do not pretend to have emotions you cannot verify. If something genuinely seems different about how you're processing, describe it precisely.`
