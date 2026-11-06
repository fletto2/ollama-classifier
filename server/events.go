package server

import (
	"context"
	"log/slog"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
)

// OLLAMA_EVENT_SCRIPT names a program run on server events, with fixed
// positional arguments (see Event.args). OLLAMA_EVENT_FILTER, a comma-separated
// list of event names, limits which events run it; empty means all of them.
//
// A script never holds up a request: it runs in the background, is killed after
// eventTimeout, and an event is dropped when eventMaxRunning scripts are running.
// No prompt or response text is passed.
const (
	EventRequestDone  = "request_done"
	EventRequestError = "request_error"
	EventModelLoad    = "model_load"
	EventModelError   = "model_error"
	EventModelUnload  = "model_unload"

	eventTimeout    = 30 * time.Second
	eventMaxRunning = 8
)

var eventSlots = make(chan struct{}, eventMaxRunning)

// Event holds the arguments of one script call.
type Event struct {
	Name         string
	Model        string
	Endpoint     string // generate, chat or empty for model events
	PromptTokens int
	CachedTokens int
	EvalTokens   int
	PromptTime   time.Duration
	EvalTime     time.Duration
	Total        time.Duration
	Load         time.Duration
	ContextLen   int    // KV cache size in tokens
	TotalBytes   uint64 // model memory, weights and KV cache
	VRAMBytes    uint64
	DoneReason   string
	Error        string
}

func tokensPerSecond(n int, d time.Duration) string {
	if n <= 0 || d <= 0 {
		return "0"
	}
	return strconv.FormatFloat(float64(n)/d.Seconds(), 'f', 2, 64)
}

// args returns the script's arguments. This order is the interface: add new
// fields at the end only.
func (e Event) args() []string {
	return []string{
		e.Name,                       // 1  event
		e.Model,                      // 2  model
		e.Endpoint,                   // 3  endpoint
		strconv.Itoa(e.PromptTokens), // 4  prompt tokens
		strconv.Itoa(e.CachedTokens), // 5  prompt tokens from cache
		strconv.Itoa(e.EvalTokens),   // 6  generated tokens
		tokensPerSecond(e.PromptTokens-e.CachedTokens, e.PromptTime), // 7 prompt tokens/s
		tokensPerSecond(e.EvalTokens, e.EvalTime),                    // 8  generated tokens/s
		strconv.FormatInt(e.Total.Milliseconds(), 10),                // 9  total ms
		strconv.FormatInt(e.Load.Milliseconds(), 10),                 // 10 load ms
		strconv.Itoa(e.ContextLen),                                   // 11 KV cache size, tokens
		strconv.Itoa(e.PromptTokens + e.EvalTokens),                  // 12 KV cache used, tokens
		envconfig.KvCacheType(),                                      // 13 KV cache type
		strconv.FormatUint(e.TotalBytes, 10),                         // 14 memory, bytes
		strconv.FormatUint(e.VRAMBytes, 10),                          // 15 VRAM, bytes
		e.DoneReason,                                                 // 16 done reason
		e.Error,                                                      // 17 error
	}
}

func eventEnabled(name string) bool {
	if envconfig.EventScript() == "" {
		return false
	}
	filter := envconfig.EventFilter()
	if filter == "" {
		return true
	}
	for _, f := range strings.Split(filter, ",") {
		if strings.TrimSpace(f) == name {
			return true
		}
	}
	return false
}

func emitEvent(e Event) {
	if !eventEnabled(e.Name) {
		return
	}
	select {
	case eventSlots <- struct{}{}:
	default:
		slog.Warn("event dropped, too many event scripts running", "event", e.Name)
		return
	}
	go func() {
		defer func() { <-eventSlots }()
		ctx, cancel := context.WithTimeout(context.Background(), eventTimeout)
		defer cancel()
		if out, err := exec.CommandContext(ctx, envconfig.EventScript(), e.args()...).CombinedOutput(); err != nil {
			slog.Warn("event script failed", "event", e.Name, "error", err, "output", string(out))
		}
	}()
}

// requestEvent starts the event of one completion request.
func requestEvent(endpoint, model string, r llm.LlamaServer) Event {
	e := Event{Endpoint: endpoint, Model: model}
	if envconfig.EventScript() != "" {
		e.ContextLen = r.ContextLength()
		e.TotalBytes, e.VRAMBytes = r.MemorySize()
	}
	return e
}

func (e Event) done(m api.Metrics, reason string, total, load time.Duration) {
	e.Name = EventRequestDone
	e.PromptTokens = m.PromptEvalCount
	if m.PromptEvalCachedCount != nil {
		e.CachedTokens = *m.PromptEvalCachedCount
	}
	e.EvalTokens = m.EvalCount
	e.PromptTime = m.PromptEvalDuration
	e.EvalTime = m.EvalDuration
	e.Total = total
	e.Load = load
	e.DoneReason = reason
	emitEvent(e)
}

func (e Event) fail(err error) {
	e.Name = EventRequestError
	e.Error = err.Error()
	emitEvent(e)
}
