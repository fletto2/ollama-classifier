package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

// Classifier is implemented by runners that serve classifier heads on the loaded model
// (llama-server --classifier, POST /classify).
type Classifier interface {
	Classify(ctx context.Context, inputs []string) ([]api.ClassifyResponse, error)
}

// LoRATrainJob is the state of a LoRA training job of llama-server (GET /lora/train).
type LoRATrainJob struct {
	ID        int     `json:"id"`
	Name      string  `json:"name"`
	Status    string  `json:"status"` // queued, running, done, error, cancelled
	Step      int64   `json:"step"`
	NSteps    int64   `json:"n_steps"`
	LossFirst float64 `json:"loss_first"`
	LossLast  float64 `json:"loss_last"`
	LossEpoch float64 `json:"loss_epoch"`
	Path      string  `json:"path"`
	AdapterID int     `json:"adapter_id"`
	Error     string  `json:"error"`
}

// LoRATrainer is implemented by runners that train LoRA adapters on the loaded model
// (llama-server --lora-train, POST /lora/train).
type LoRATrainer interface {
	StartLoRATrain(ctx context.Context, body map[string]any) (LoRATrainJob, error)
	LoRATrainJobs(ctx context.Context) ([]LoRATrainJob, error)
	CancelLoRATrain(ctx context.Context, id int) error
}

var (
	_ Classifier  = (*llamaServerRunner)(nil)
	_ LoRATrainer = (*llamaServerRunner)(nil)
)

// serverJSON sends a JSON request to llama-server and decodes the JSON response.
func (s *llamaServerRunner) serverJSON(ctx context.Context, method, path string, input, output any) error {
	var body io.Reader
	if input != nil {
		data, err := json.Marshal(input)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, fmt.Sprintf("http://127.0.0.1:%d%s", s.port, path), body)
	if err != nil {
		return err
	}
	if input != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	res, err := s.httpClient().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if msg := s.lastErrMsg(); msg != "" {
			return fmt.Errorf("%s %s failed: %s: %w", method, path, msg, err)
		}
		return err
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(data)}
	}
	if err := json.Unmarshal(data, output); err != nil {
		return fmt.Errorf("invalid response from %s: %w", path, err)
	}
	return nil
}

func (s *llamaServerRunner) Classify(ctx context.Context, inputs []string) ([]api.ClassifyResponse, error) {
	if len(inputs) == 0 {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "input must not be empty"}
	}
	var out []api.ClassifyResponse
	if err := s.serverJSON(ctx, http.MethodPost, "/classify", map[string]any{"input": inputs}, &out); err != nil {
		return nil, err
	}
	if len(out) != len(inputs) {
		return nil, fmt.Errorf("classifier returned %d results for %d inputs", len(out), len(inputs))
	}
	return out, nil
}

func (s *llamaServerRunner) StartLoRATrain(ctx context.Context, body map[string]any) (LoRATrainJob, error) {
	var job LoRATrainJob
	err := s.serverJSON(ctx, http.MethodPost, "/lora/train", body, &job)
	return job, err
}

func (s *llamaServerRunner) LoRATrainJobs(ctx context.Context) ([]LoRATrainJob, error) {
	var jobs []LoRATrainJob
	err := s.serverJSON(ctx, http.MethodGet, "/lora/train", nil, &jobs)
	return jobs, err
}

func (s *llamaServerRunner) CancelLoRATrain(ctx context.Context, id int) error {
	var out map[string]any
	return s.serverJSON(ctx, http.MethodPost, "/lora/train/cancel", map[string]any{"id": id}, &out)
}
