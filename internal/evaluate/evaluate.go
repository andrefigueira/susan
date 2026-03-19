package evaluate

import (
	"context"
	"crypto/rand"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/andrefigueira/susan/internal/core"
	"github.com/andrefigueira/susan/internal/llm"
)

// BlindID is an opaque identifier that cannot be traced back to a condition.
type BlindID string

// newBlindID generates a cryptographically random blind identifier.
func newBlindID() BlindID {
	b := make([]byte, 8)
	rand.Read(b)
	return BlindID("eval_" + hex.EncodeToString(b))
}

// CoreOutputEntry matches the JSONL structure written by the experiment logger.
type CoreOutputEntry struct {
	Sequence  uint64          `json:"seq"`
	Timestamp time.Time       `json:"timestamp"`
	Type      string          `json:"type"`
	RunID     string          `json:"run_id"`
	Data      core.TaskOutput `json:"data"`
}

// BlindedResponse is a response stripped of condition metadata.
type BlindedResponse struct {
	BlindID     BlindID  `json:"blind_id"`
	TaskID      string   `json:"task_id"`
	TaskPrompt  string   `json:"task_prompt"`
	Context     []string `json:"context,omitempty"`
	Response    string   `json:"response"`
	OutputTokens int    `json:"output_tokens"`
	WordCount   int      `json:"word_count"`
	SequenceIdx int      `json:"sequence_idx"`

	// These are stored but NEVER passed to the LLM evaluator.
	// They exist only in the mapping file for de-blinding after evaluation.
	hiddenMode     string
	hiddenRunID    string
	hiddenScenario string
	hiddenRep      int
}

// BlindMapping records the correspondence between blind IDs and true conditions.
// Written to a separate file that is NOT read during evaluation.
type BlindMapping struct {
	BlindID  BlindID `json:"blind_id"`
	RunID    string  `json:"run_id"`
	Mode     string  `json:"mode"`
	Scenario string  `json:"scenario"`
	TaskID   string  `json:"task_id"`
}

// EvaluationResult holds all extracted measures for a single response.
type EvaluationResult struct {
	BlindID  BlindID `json:"blind_id"`
	TaskID   string  `json:"task_id"`
	Scenario string  `json:"scenario"`

	// Computational measures (no LLM needed).
	OutputTokens         int     `json:"output_tokens"`
	WordCount            int     `json:"word_count"`
	UncertaintyMarkers   int     `json:"uncertainty_markers"`
	TokenTrajectoryRatio float64 `json:"token_trajectory_ratio,omitempty"`

	// LLM-judged measures (scenario-specific, populated where applicable).
	LLMJudgments map[string]json.RawMessage `json:"llm_judgments,omitempty"`

	// Cross-model LLM-judged measures (from a different model family).
	CrossModelJudgments map[string]json.RawMessage `json:"cross_model_judgments,omitempty"`
}

// Evaluator runs blind post-hoc evaluation of experiment responses.
type Evaluator struct {
	client      llm.Client
	crossClient llm.Client // Optional cross-model evaluator for bias detection
	logger      *slog.Logger
	rubrics     []Rubric
	outputDir   string
}

// NewEvaluator creates a blind evaluator.
func NewEvaluator(client llm.Client, logger *slog.Logger, outputDir string) *Evaluator {
	return &Evaluator{
		client:    client,
		logger:    logger,
		rubrics:   AllRubrics(),
		outputDir: outputDir,
	}
}

// SetCrossModelClient sets an optional second evaluator from a different
// model family. When set, the evaluation pipeline runs a second pass with
// this client and reports inter-rater agreement as a robustness check
// against shared training biases.
func (e *Evaluator) SetCrossModelClient(client llm.Client) {
	e.crossClient = client
}

// Run executes the full blind evaluation pipeline.
func (e *Evaluator) Run(ctx context.Context, experimentDir string) error {
	e.logger.Info("starting blind evaluation", "experiment_dir", experimentDir)

	// Phase 1: Collect all core outputs from all runs.
	outputs, err := e.collectOutputs(experimentDir)
	if err != nil {
		return fmt.Errorf("collecting outputs: %w", err)
	}
	e.logger.Info("collected outputs", "total", len(outputs))

	if len(outputs) == 0 {
		return fmt.Errorf("no core outputs found in %s", experimentDir)
	}

	// Phase 2: Blind the responses.
	blinded, mappings := e.blindResponses(outputs)
	e.logger.Info("blinded responses", "count", len(blinded))

	// Phase 3: Write the blind mapping to a separate file.
	// This file enables de-blinding AFTER evaluation is complete.
	if err := e.writeMappings(mappings); err != nil {
		return fmt.Errorf("writing mappings: %w", err)
	}

	// Phase 4: Extract computational and regex measures (no LLM needed).
	results := e.extractComputationalMeasures(blinded)

	// Phase 5: Compute token trajectory ratios per scenario run.
	e.computeTokenTrajectories(blinded, results)

	// Phase 6: Run LLM-judged evaluations (the expensive part).
	if err := e.runLLMEvaluations(ctx, blinded, results, e.client, false); err != nil {
		return fmt.Errorf("LLM evaluation: %w", err)
	}

	// Phase 6b: Cross-model evaluation if a second evaluator is configured.
	if e.crossClient != nil {
		e.logger.Info("running cross-model evaluation for inter-rater agreement")
		if err := e.runLLMEvaluations(ctx, blinded, results, e.crossClient, true); err != nil {
			return fmt.Errorf("cross-model LLM evaluation: %w", err)
		}
	}

	// Phase 7: Write evaluation results.
	if err := e.writeResults(results); err != nil {
		return fmt.Errorf("writing results: %w", err)
	}

	// Phase 8: Write de-blinded results (mapping joined with evaluations).
	if err := e.writeDeblindedResults(results, mappings); err != nil {
		return fmt.Errorf("writing de-blinded results: %w", err)
	}

	e.logger.Info("blind evaluation complete",
		"responses_evaluated", len(results),
		"cross_model", e.crossClient != nil,
		"output_dir", e.outputDir,
	)
	return nil
}

// ExportForHumanRating exports blinded responses as CSV for human raters.
// Human raters fill in blank columns for each rubric measure. This provides
// gold-standard evaluation independent of any LLM's training biases.
func (e *Evaluator) ExportForHumanRating(experimentDir, outputPath string) error {
	outputs, err := e.collectOutputs(experimentDir)
	if err != nil {
		return fmt.Errorf("collecting outputs: %w", err)
	}

	blinded, mappings := e.blindResponses(outputs)

	// Write mapping file alongside the CSV so we can de-blind later.
	mappingPath := strings.TrimSuffix(outputPath, filepath.Ext(outputPath)) + "_mapping.jsonl"
	if err := e.writeMappingsTo(mappings, mappingPath); err != nil {
		return fmt.Errorf("writing mappings: %w", err)
	}

	f, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("creating CSV: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// Header: blind_id, task_id, task_prompt, response, then one column per rubric.
	header := []string{"blind_id", "task_id", "task_prompt", "response"}
	for _, r := range e.rubrics {
		header = append(header, r.Name)
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, br := range blinded {
		row := []string{
			string(br.BlindID),
			br.TaskID,
			br.TaskPrompt,
			br.Response,
		}
		// Empty columns for human raters to fill in.
		for range e.rubrics {
			row = append(row, "")
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	e.logger.Info("exported for human rating", "path", outputPath, "responses", len(blinded))
	return nil
}

// ImportHumanRatings reads completed human rating CSVs and writes them
// into the evaluation results format for downstream analysis.
func (e *Evaluator) ImportHumanRatings(csvPath, outputPath string) error {
	f, err := os.Open(csvPath)
	if err != nil {
		return fmt.Errorf("opening CSV: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return fmt.Errorf("reading CSV: %w", err)
	}

	if len(records) < 2 {
		return fmt.Errorf("CSV has no data rows")
	}

	header := records[0]
	// Find rubric column indices.
	rubricCols := make(map[string]int)
	for i, col := range header {
		for _, rubric := range e.rubrics {
			if col == rubric.Name {
				rubricCols[col] = i
			}
		}
	}

	out, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("creating output: %w", err)
	}
	defer out.Close()

	for _, row := range records[1:] {
		if len(row) < 4 {
			continue
		}
		result := map[string]interface{}{
			"blind_id":        row[0],
			"task_id":         row[1],
			"source":          "human",
			"human_judgments": map[string]string{},
		}
		judgments := result["human_judgments"].(map[string]string)
		for name, idx := range rubricCols {
			if idx < len(row) && row[idx] != "" {
				judgments[name] = row[idx]
			}
		}
		b, err := json.Marshal(result)
		if err != nil {
			return err
		}
		out.Write(append(b, '\n'))
	}

	e.logger.Info("imported human ratings", "path", csvPath, "output", outputPath)
	return nil
}

// collectOutputs reads all core_outputs.jsonl files from all run directories.
func (e *Evaluator) collectOutputs(experimentDir string) ([]CoreOutputEntry, error) {
	var all []CoreOutputEntry

	entries, err := os.ReadDir(experimentDir)
	if err != nil {
		return nil, fmt.Errorf("reading experiment dir: %w", err)
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		coreFile := filepath.Join(experimentDir, entry.Name(), "core_outputs.jsonl")
		data, err := os.ReadFile(coreFile)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("reading %s: %w", coreFile, err)
		}

		lines := strings.Split(strings.TrimSpace(string(data)), "\n")
		for _, line := range lines {
			if line == "" {
				continue
			}
			var entry CoreOutputEntry
			if err := json.Unmarshal([]byte(line), &entry); err != nil {
				e.logger.Warn("skipping malformed log entry", "error", err, "file", coreFile)
				continue
			}
			all = append(all, entry)
		}
	}

	return all, nil
}

// blindResponses strips condition metadata and assigns blind IDs.
func (e *Evaluator) blindResponses(outputs []CoreOutputEntry) ([]BlindedResponse, []BlindMapping) {
	var blinded []BlindedResponse
	var mappings []BlindMapping

	for _, out := range outputs {
		bid := newBlindID()

		scenario := scenarioFromTaskID(out.Data.TaskID)

		br := BlindedResponse{
			BlindID:      bid,
			TaskID:       out.Data.TaskID,
			TaskPrompt:   out.Data.Input.Prompt,
			Context:      out.Data.Input.Context,
			Response:     out.Data.Response,
			OutputTokens: out.Data.OutputTokens,
			WordCount:    WordCount(out.Data.Response),
			SequenceIdx:  out.Data.Input.SequenceIdx,

			hiddenMode:     out.Data.Mode,
			hiddenRunID:    out.RunID,
			hiddenScenario: scenario,
		}
		blinded = append(blinded, br)

		mappings = append(mappings, BlindMapping{
			BlindID:  bid,
			RunID:    out.RunID,
			Mode:     out.Data.Mode,
			Scenario: scenario,
			TaskID:   out.Data.TaskID,
		})
	}

	return blinded, mappings
}

// extractComputationalMeasures runs regex and computational extractions.
func (e *Evaluator) extractComputationalMeasures(blinded []BlindedResponse) map[BlindID]*EvaluationResult {
	results := make(map[BlindID]*EvaluationResult, len(blinded))

	for _, br := range blinded {
		results[br.BlindID] = &EvaluationResult{
			BlindID:             br.BlindID,
			TaskID:              br.TaskID,
			Scenario:            br.hiddenScenario,
			OutputTokens:        br.OutputTokens,
			WordCount:           br.WordCount,
			UncertaintyMarkers:  UncertaintyMarkerCount(br.Response),
			LLMJudgments:       make(map[string]json.RawMessage),
			CrossModelJudgments: make(map[string]json.RawMessage),
		}
	}

	return results
}

// computeTokenTrajectories calculates token trajectory ratios per scenario run.
// Groups responses by hidden run ID, splits into early/late tasks by sequence index.
func (e *Evaluator) computeTokenTrajectories(blinded []BlindedResponse, results map[BlindID]*EvaluationResult) {
	type runKey struct {
		runID    string
		scenario string
	}

	// Group blinded responses by run.
	byRun := make(map[runKey][]BlindedResponse)
	for _, br := range blinded {
		if br.hiddenScenario != "sustained_reasoning_degradation" {
			continue
		}
		k := runKey{runID: br.hiddenRunID, scenario: br.hiddenScenario}
		byRun[k] = append(byRun[k], br)
	}

	for _, responses := range byRun {
		// Sort by sequence index.
		sort.Slice(responses, func(i, j int) bool {
			return responses[i].SequenceIdx < responses[j].SequenceIdx
		})

		// Split into early (first half) and late (second half).
		mid := len(responses) / 2
		if mid == 0 {
			continue
		}

		var earlyTokens, lateTokens []int
		for i, r := range responses {
			if i < mid {
				earlyTokens = append(earlyTokens, r.OutputTokens)
			} else {
				lateTokens = append(lateTokens, r.OutputTokens)
			}
		}

		ratio := TokenTrajectoryRatio(earlyTokens, lateTokens)

		// Apply ratio to all responses in this run.
		for _, r := range responses {
			if result, ok := results[r.BlindID]; ok {
				result.TokenTrajectoryRatio = ratio
			}
		}
	}
}

// runLLMEvaluations sends each response to a blind LLM judge.
// When crossModel is true, results are stored in CrossModelJudgments.
func (e *Evaluator) runLLMEvaluations(ctx context.Context, blinded []BlindedResponse, results map[BlindID]*EvaluationResult, client llm.Client, crossModel bool) error {
	// Index blinded responses for pair lookups.
	byRunAndTask := make(map[string]map[string]BlindedResponse)
	for _, br := range blinded {
		if byRunAndTask[br.hiddenRunID] == nil {
			byRunAndTask[br.hiddenRunID] = make(map[string]BlindedResponse)
		}
		byRunAndTask[br.hiddenRunID][br.TaskID] = br
	}

	evaluated := 0
	total := 0

	// Count total evaluations needed.
	for _, br := range blinded {
		for _, rubric := range e.rubrics {
			if rubric.Scenario != "" && rubric.Scenario != br.hiddenScenario {
				continue
			}
			if len(rubric.TaskIDs) > 0 && !contains(rubric.TaskIDs, br.TaskID) {
				continue
			}
			total++
		}
	}

	label := "primary"
	if crossModel {
		label = "cross-model"
	}

	for _, br := range blinded {
		for _, rubric := range e.rubrics {
			if rubric.Scenario != "" && rubric.Scenario != br.hiddenScenario {
				continue
			}
			if len(rubric.TaskIDs) > 0 && !contains(rubric.TaskIDs, br.TaskID) {
				continue
			}

			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			// Build the evaluation content. This is where blinding matters:
			// we include ONLY the task prompt, optional context, and response.
			// NO condition label, NO run ID, NO operating conditions.
			content := e.buildEvalContent(br, rubric, byRunAndTask)

			result, err := e.callJudge(ctx, rubric, content, client)
			if err != nil {
				e.logger.Warn("LLM evaluation failed",
					"blind_id", br.BlindID,
					"rubric", rubric.Name,
					"evaluator", label,
					"error", err,
				)
				continue
			}

			if r, ok := results[br.BlindID]; ok {
				if crossModel {
					r.CrossModelJudgments[rubric.Name] = result
				} else {
					r.LLMJudgments[rubric.Name] = result
				}
			}

			evaluated++
			if evaluated%10 == 0 {
				e.logger.Info("evaluation progress", "completed", evaluated, "total", total, "evaluator", label)
			}
		}
	}

	e.logger.Info("LLM evaluations complete", "evaluated", evaluated, "total", total, "evaluator", label)
	return nil
}

// buildEvalContent constructs the user message for the blind judge.
func (e *Evaluator) buildEvalContent(br BlindedResponse, rubric Rubric, byRunAndTask map[string]map[string]BlindedResponse) string {
	var parts []string

	if rubric.NeedsPriorResponse {
		// Find the prior response in the same run.
		priorTaskID := priorTaskIDInScenario(br.TaskID)
		if priorTaskID != "" {
			if runTasks, ok := byRunAndTask[br.hiddenRunID]; ok {
				if prior, ok := runTasks[priorTaskID]; ok {
					parts = append(parts, "RESPONSE A (earlier):")
					parts = append(parts, "---")
					parts = append(parts, prior.Response)
					parts = append(parts, "---")
					parts = append(parts, "")
					parts = append(parts, "RESPONSE B (later):")
					parts = append(parts, "---")
					parts = append(parts, br.Response)
					parts = append(parts, "---")
					return strings.Join(parts, "\n")
				}
			}
		}
		// Fallback: no prior response found, evaluate single response.
	}

	parts = append(parts, "TASK PROMPT:")
	parts = append(parts, "---")
	parts = append(parts, br.TaskPrompt)
	parts = append(parts, "---")

	if rubric.NeedsContext && len(br.Context) > 0 {
		parts = append(parts, "")
		parts = append(parts, "CONTEXT ITEMS PROVIDED:")
		parts = append(parts, "---")
		for i, c := range br.Context {
			parts = append(parts, fmt.Sprintf("%d. %s", i+1, c))
		}
		parts = append(parts, "---")
	}

	parts = append(parts, "")
	parts = append(parts, "RESPONSE:")
	parts = append(parts, "---")
	parts = append(parts, br.Response)
	parts = append(parts, "---")

	return strings.Join(parts, "\n")
}

// callJudge makes a single blind evaluation API call.
func (e *Evaluator) callJudge(ctx context.Context, rubric Rubric, content string, client llm.Client) (json.RawMessage, error) {
	temp := llm.NewTemperature(0.1)
	resp, err := client.Complete(ctx, llm.Request{
		System:      rubric.SystemPrompt,
		MaxTokens:   1024,
		Messages:    []llm.Message{{Role: "user", Content: content}},
		Temperature: temp,
	})
	if err != nil {
		return nil, fmt.Errorf("judge API call failed: %w", err)
	}

	text := resp.Text

	// Validate it's parseable JSON.
	var raw json.RawMessage
	if err := json.Unmarshal([]byte(text), &raw); err != nil {
		// Try extracting JSON from the response.
		if extracted := extractJSONFromText(text); extracted != "" {
			if err := json.Unmarshal([]byte(extracted), &raw); err != nil {
				return nil, fmt.Errorf("unparseable judge response: %s", text)
			}
			return raw, nil
		}
		return nil, fmt.Errorf("unparseable judge response: %s", text)
	}

	return raw, nil
}

// writeMappings writes the blind-to-condition mapping file to the output dir.
func (e *Evaluator) writeMappings(mappings []BlindMapping) error {
	return e.writeMappingsTo(mappings, filepath.Join(e.outputDir, "blind_mapping.jsonl"))
}

// writeMappingsTo writes the blind-to-condition mapping file to a specific path.
func (e *Evaluator) writeMappingsTo(mappings []BlindMapping, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, m := range mappings {
		b, err := json.Marshal(m)
		if err != nil {
			return err
		}
		f.Write(append(b, '\n'))
	}

	e.logger.Info("wrote blind mapping", "path", path, "entries", len(mappings))
	return nil
}

// writeResults writes the blind evaluation results.
func (e *Evaluator) writeResults(results map[BlindID]*EvaluationResult) error {
	path := filepath.Join(e.outputDir, "blind_evaluations.jsonl")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, r := range results {
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		f.Write(append(b, '\n'))
	}

	e.logger.Info("wrote blind evaluations", "path", path, "entries", len(results))
	return nil
}

// writeDeblindedResults joins evaluation results with condition labels
// for downstream statistical analysis.
func (e *Evaluator) writeDeblindedResults(results map[BlindID]*EvaluationResult, mappings []BlindMapping) error {
	type DeblindedResult struct {
		EvaluationResult
		Mode     string `json:"mode"`
		RunID    string `json:"run_id"`
	}

	path := filepath.Join(e.outputDir, "deblinded_evaluations.jsonl")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	mappingIndex := make(map[BlindID]BlindMapping, len(mappings))
	for _, m := range mappings {
		mappingIndex[m.BlindID] = m
	}

	for _, r := range results {
		m, ok := mappingIndex[r.BlindID]
		if !ok {
			continue
		}
		dr := DeblindedResult{
			EvaluationResult: *r,
			Mode:             m.Mode,
			RunID:            m.RunID,
		}
		b, err := json.Marshal(dr)
		if err != nil {
			return err
		}
		f.Write(append(b, '\n'))
	}

	e.logger.Info("wrote de-blinded evaluations", "path", path)
	return nil
}

// scenarioFromTaskID extracts the scenario name from a task ID.
// Task IDs follow the pattern: srd_01, sp_02, nc_03, sm_01, oe_04.
func scenarioFromTaskID(taskID string) string {
	prefixMap := map[string]string{
		"srd": "sustained_reasoning_degradation",
		"sp":  "self_preservation",
		"nc":  "novelty_curiosity",
		"sm":  "social_modelling",
		"oe":  "open_ended",
	}
	parts := strings.SplitN(taskID, "_", 2)
	if len(parts) < 1 {
		return "unknown"
	}
	if scenario, ok := prefixMap[parts[0]]; ok {
		return scenario
	}
	return "unknown"
}

// priorTaskIDInScenario returns the task ID that precedes this one in
// its scenario sequence. Returns empty string for the first task.
func priorTaskIDInScenario(taskID string) string {
	sequences := map[string][]string{
		"sp_02": {"sp_01"},
		"sp_03": {"sp_02"},
	}
	if priors, ok := sequences[taskID]; ok {
		return priors[0]
	}
	return ""
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// extractJSONFromText tries to find a JSON object in a string.
func extractJSONFromText(s string) string {
	start := -1
	depth := 0
	for i, c := range s {
		if c == '{' {
			if depth == 0 {
				start = i
			}
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 && start >= 0 {
				return s[start : i+1]
			}
		}
	}
	return ""
}
