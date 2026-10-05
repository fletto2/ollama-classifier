package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// resolveLocalGGUFModel resolves a request's model name to a local GGUF model, writing the error
// response itself when it fails.
func resolveLocalGGUFModel(c *gin.Context, requested string) (*Model, model.Name, bool) {
	ref, err := parseAndValidateModelRef(requested)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", requested))
		return nil, model.Name{}, false
	}
	if ref.Source == modelSourceCloud {
		c.JSON(http.StatusBadRequest, gin.H{"error": "this endpoint requires a local model"})
		return nil, model.Name{}, false
	}
	name, err := getExistingName(ref.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", requested)})
		return nil, model.Name{}, false
	}
	m, err := GetModel(name.String())
	if err != nil {
		handleScheduleError(c, requested, err)
		return nil, model.Name{}, false
	}
	if !m.isGGUF() {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model %q is not a GGUF model", requested)})
		return nil, model.Name{}, false
	}
	return m, name, true
}

// ClassifyHandler runs the classifier heads of a model (CLASSIFIER in its Modelfile) on the
// loaded model; one typed-decision response per input (an object for a string input, an array
// for a list of strings).
func (s *Server) ClassifyHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<20)
	var req api.ClassifyRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		var sizeErr *http.MaxBytesError
		if errors.As(err, &sizeErr) {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body must not exceed 32 MiB"})
			return
		}
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var inputs []string
	single := false
	switch in := req.Input.(type) {
	case string:
		inputs, single = []string{in}, true
	case []any:
		for _, v := range in {
			s, ok := v.(string)
			if !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "input must be a string or a list of strings"})
				return
			}
			inputs = append(inputs, s)
		}
	default:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "input must be a string or a list of strings"})
		return
	}
	if len(inputs) == 0 || slices.Contains(inputs, "") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "input must not be empty"})
		return
	}

	m, _, ok := resolveLocalGGUFModel(c, req.Model)
	if !ok {
		return
	}
	if len(m.ClassifierPaths) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model %q has no classifier heads; add CLASSIFIER <head.gguf> to its Modelfile", req.Model)})
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), m, []model.Capability{}, nil, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	classifier, ok := r.(llm.Classifier)
	if !ok {
		c.JSON(http.StatusNotImplemented, gin.H{"error": fmt.Sprintf("the runner of model %q does not support classifier heads", req.Model)})
		return
	}
	results, err := classifier.Classify(c.Request.Context(), inputs)
	if err != nil {
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}
	for i := range results {
		results[i].Model = req.Model
	}
	if single {
		c.JSON(http.StatusOK, results[0])
		return
	}
	c.JSON(http.StatusOK, results)
}

// TrainHandler trains a LoRA adapter on the loaded base model (llama-server --lora-train, enabled
// with OLLAMA_LORA_TRAIN=1), streams the progress, and saves the base model's layers plus the
// adapter as a new model. The base model must not have adapters of its own: training runs on the
// bare base weights.
func (s *Server) TrainHandler(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 64<<20)
	var req api.TrainRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		var sizeErr *http.MaxBytesError
		if errors.As(err, &sizeErr) {
			c.AbortWithStatusJSON(http.StatusRequestEntityTooLarge, gin.H{"error": "request body must not exceed 64 MiB"})
			return
		}
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if !envconfig.LoRATrain() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "LoRA training is disabled; start the server with OLLAMA_LORA_TRAIN=1"})
		return
	}
	if req.Text == "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "text is required"})
		return
	}
	newName := model.ParseName(req.Name)
	if !newName.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("invalid model name %q", req.Name)})
		return
	}

	m, baseName, ok := resolveLocalGGUFModel(c, req.Model)
	if !ok {
		return
	}
	if newName.EqualFold(baseName) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "the new model must have a different name than the base model"})
		return
	}
	if len(m.AdapterPaths) > 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("model %q already has LoRA adapters; train from a model without adapters", req.Model)})
		return
	}

	// the new model is built from the base model as it is now, not as it may be after training
	baseLayers, baseConfig, err := parseFromModel(c.Request.Context(), baseName, func(api.ProgressResponse) {})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	r, _, _, err := s.scheduleRunner(c.Request.Context(), m, []model.Capability{}, nil, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}
	trainer, ok := r.(llm.LoRATrainer)
	if !ok {
		c.JSON(http.StatusNotImplemented, gin.H{"error": fmt.Sprintf("the runner of model %q does not support LoRA training", req.Model)})
		return
	}

	// the runner writes the adapter to its --lora-train-dir; it is not registered with the runner
	file := fmt.Sprintf("ollama-%d.gguf", time.Now().UnixNano())
	adapterPath := filepath.Join(llm.LoRATrainDir(), file)
	body := map[string]any{
		"text":     req.Text,
		"name":     req.Name,
		"file":     file,
		"register": false,
		"rank":     cmpOr(req.Rank, 8),
		"lr":       cmpOr(req.LearningRate, 1e-4),
		"epochs":   cmpOr(req.Epochs, 1),
		"n_ctx":    cmpOr(req.NumCtx, 256),
		"priority": cmpOr(req.Priority, "idle"),
	}
	if req.Alpha > 0 {
		body["alpha"] = req.Alpha
	}
	if req.Targets != "" {
		body["targets"] = req.Targets
	}
	if req.Seed != nil {
		body["seed"] = *req.Seed
	}

	ctx := c.Request.Context()
	ch := make(chan any)
	go func() {
		defer close(ch)
		defer os.Remove(adapterPath)
		send := func(resp any) bool {
			select {
			case ch <- resp:
				return true
			case <-ctx.Done():
				return false
			}
		}
		fail := func(err error) {
			status := http.StatusInternalServerError
			var statusErr api.StatusError
			if errors.As(err, &statusErr) {
				status = statusErr.StatusCode
			}
			send(gin.H{"error": err.Error(), "status": status})
		}

		job, err := trainer.StartLoRATrain(ctx, body)
		if err != nil {
			fail(err)
			return
		}
		// stop the job on any early return (client gone, polling error); a finished job ignores the cancel
		finished := false
		defer func() {
			if !finished {
				cctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				defer cancel()
				_ = trainer.CancelLoRATrain(cctx, job.ID)
			}
		}()

		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for job.Status == "queued" || job.Status == "running" {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			jobs, err := trainer.LoRATrainJobs(ctx)
			if err != nil {
				if ctx.Err() == nil {
					fail(err)
				}
				return
			}
			for _, j := range jobs {
				if j.ID == job.ID {
					job = j
				}
			}
			status := "training"
			if job.LossLast > 0 {
				status = fmt.Sprintf("training (loss %.3f)", job.LossLast)
			}
			send(api.ProgressResponse{Status: status, Total: job.NSteps, Completed: job.Step})
		}
		finished = true
		if job.Status != "done" {
			fail(fmt.Errorf("training %s: %s", job.Status, job.Error))
			return
		}

		fn := func(resp api.ProgressResponse) { send(resp) }
		fn(api.ProgressResponse{Status: fmt.Sprintf("trained %d steps, loss %.3f -> %.3f", job.NSteps, job.LossFirst, job.LossLast)})

		f, err := os.Open(adapterPath)
		if err != nil {
			fail(err)
			return
		}
		layer, err := manifest.NewLayer(f, manifest.MediaTypeImageAdapter)
		f.Close()
		if err != nil {
			fail(err)
			return
		}
		layers := append(slices.Clone(baseLayers), &modelLayer{Layer: layer})
		config := baseConfig
		if err := createModel(ctx, api.CreateRequest{Model: newName.String()}, newName, layers, &config, fn); err != nil {
			fail(err)
			return
		}
		send(api.ProgressResponse{Status: "success"})
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}
	streamResponse(c, ch)
}

func cmpOr[T comparable](v, def T) T {
	var zero T
	if v == zero {
		return def
	}
	return v
}
