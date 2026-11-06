package server

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func eventScript(t *testing.T, out string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "ev.sh")
	body := fmt.Sprintf("#!/bin/sh\nfor a in \"$@\"; do printf '%%s|' \"$a\"; done > %s.tmp\nmv %s.tmp %s\n", out, out, out)
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func waitFile(p string) string {
	for range 200 {
		if b, err := os.ReadFile(p); err == nil {
			return string(b)
		}
		time.Sleep(10 * time.Millisecond)
	}
	return ""
}

func TestEventArgs(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out")
	t.Setenv("OLLAMA_EVENT_SCRIPT", eventScript(t, out))
	t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
	e := Event{Model: "m", Endpoint: "chat", ContextLen: 4096, TotalBytes: 7, VRAMBytes: 5}
	cached := 40
	e.done(api.Metrics{PromptEvalCount: 100, PromptEvalCachedCount: &cached, PromptEvalDuration: 2 * time.Second, EvalCount: 50, EvalDuration: time.Second},
		"stop", 3*time.Second, 500*time.Millisecond)
	want := "request_done|m|chat|100|40|50|30.00|50.00|3000|500|4096|150|q8_0|7|5|stop||"
	if got := waitFile(out); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestEventFilter(t *testing.T) {
	out := filepath.Join(t.TempDir(), "out")
	t.Setenv("OLLAMA_EVENT_SCRIPT", eventScript(t, out))
	t.Setenv("OLLAMA_EVENT_FILTER", "request_error, model_load")
	Event{Model: "m"}.done(api.Metrics{}, "stop", 0, 0)
	Event{Model: "m"}.fail(fmt.Errorf("boom"))
	got := waitFile(out)
	if !strings.HasPrefix(got, "request_error|") || !strings.Contains(got, "|boom|") {
		t.Fatalf("got %q", got)
	}
}

func TestEventOffAndFull(t *testing.T) {
	t.Setenv("OLLAMA_EVENT_SCRIPT", "")
	if eventEnabled(EventRequestDone) {
		t.Fatal("enabled without a script")
	}
	t.Setenv("OLLAMA_EVENT_SCRIPT", "/bin/true")
	for range eventMaxRunning {
		eventSlots <- struct{}{}
	}
	defer func() {
		for range eventMaxRunning {
			<-eventSlots
		}
	}()
	done := make(chan struct{})
	go func() { emitEvent(Event{Name: EventRequestDone}); close(done) }()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("emitEvent blocked with all slots busy")
	}
}
