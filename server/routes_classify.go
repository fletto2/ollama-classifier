package server

import (
	"context"
	"encoding/json"
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

// classifierProxy forwards a request body to one of the runner's llama-server classifier endpoints. The model
// is resolved and scheduled as usual; "model" and "keep_alive" are consumed here, every other field reaches
// llama-server unchanged, and the "model" of each answer is set to the requested name.
func (s *Server) classifierProxy(c *gin.Context, path string, check func(m *Model, body map[string]any) string) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, 32<<20)
	var body map[string]any
	if err := json.NewDecoder(c.Request.Body).Decode(&body); errors.Is(err, io.EOF) {
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
	name, _ := body["model"].(string)
	var keepAlive *api.Duration
	if v, ok := body["keep_alive"]; ok {
		raw, _ := json.Marshal(v)
		var d api.Duration
		if err := json.Unmarshal(raw, &d); err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid keep_alive: " + err.Error()})
			return
		}
		keepAlive = &d
	}
	delete(body, "model")
	delete(body, "keep_alive")

	m, _, ok := resolveLocalGGUFModel(c, name)
	if !ok {
		return
	}
	if msg := check(m, body); msg != "" {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": msg})
		return
	}
	r, _, _, err := s.scheduleRunner(c.Request.Context(), m, []model.Capability{}, nil, keepAlive, nil)
	if err != nil {
		handleScheduleError(c, name, err)
		return
	}
	proxy, ok := r.(llm.ClassifierProxy)
	if !ok {
		c.JSON(http.StatusNotImplemented, gin.H{"error": fmt.Sprintf("the runner of model %q does not serve %s", name, path)})
		return
	}
	out, err := proxy.ClassifierPost(c.Request.Context(), path, body)
	if err != nil {
		status := http.StatusInternalServerError
		var statusErr api.StatusError
		if errors.As(err, &statusErr) {
			status = statusErr.StatusCode
		}
		c.JSON(status, gin.H{"error": err.Error()})
		return
	}
	var v any
	if err := json.Unmarshal(out, &v); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "invalid answer from the runner: " + err.Error()})
		return
	}
	setModel := func(x any) {
		if o, ok := x.(map[string]any); ok {
			if _, has := o["model"]; has {
				o["model"] = name
			}
		}
	}
	if arr, ok := v.([]any); ok {
		for _, x := range arr {
			setModel(x)
		}
	} else {
		setModel(v)
	}
	c.JSON(http.StatusOK, v)
}

// ClassifyHandler answers the model's classifier heads (CLASSIFIER in its Modelfile) and/or questions given in
// the request ("questions", answered by similarity to the answers' descriptions; needs OLLAMA_CLASSIFIER_FEATURES).
func (s *Server) ClassifyHandler(c *gin.Context) {
	s.classifierProxy(c, "/classify", func(m *Model, body map[string]any) string {
		if body["input"] == nil {
			return "input is required"
		}
		if _, q := body["questions"]; q {
			if !envconfig.ClassifierFeatures() && !envconfig.ClassifierTrain() {
				return "questions in the request need OLLAMA_CLASSIFIER_FEATURES=1"
			}
			return ""
		}
		if len(m.ClassifierPaths) == 0 {
			return "model has no classifier heads; add CLASSIFIER <head.gguf> to its Modelfile, or send \"questions\""
		}
		return ""
	})
}

// FeaturesHandler returns pooled hidden states of chosen layers (needs OLLAMA_CLASSIFIER_FEATURES).
func (s *Server) FeaturesHandler(c *gin.Context) {
	s.classifierProxy(c, "/features", func(m *Model, body map[string]any) string {
		if !envconfig.ClassifierFeatures() && !envconfig.ClassifierTrain() {
			return "features are disabled; start the server with OLLAMA_CLASSIFIER_FEATURES=1"
		}
		return ""
	})
}

// OptionsHandler scores fixed answers after a prompt by their log-probabilities (needs OLLAMA_OPTIONS).
func (s *Server) OptionsHandler(c *gin.Context) {
	s.classifierProxy(c, "/options", func(m *Model, body map[string]any) string {
		if !envconfig.Options() {
			return "options are disabled; start the server with OLLAMA_OPTIONS=1"
		}
		return ""
	})
}

// ClassifyTrainHandler fits a classifier head on labelled inputs and registers it with the running model
// (needs OLLAMA_CLASSIFIER_TRAIN; heads requested with "save": true go to <models>/classifier-train).
func (s *Server) ClassifyTrainHandler(c *gin.Context) {
	s.classifierProxy(c, "/classify/train", func(m *Model, body map[string]any) string {
		if !envconfig.ClassifierTrain() {
			return "classifier training is disabled; start the server with OLLAMA_CLASSIFIER_TRAIN=1"
		}
		return ""
	})
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
