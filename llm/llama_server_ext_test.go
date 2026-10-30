package llm

import (
	"slices"
	"testing"
)

func TestAppendClassifierArgsMoE(t *testing.T) {
	got := appendClassifierArgs(nil, LlamaServerConfig{MoESelectionBias: "/m/bias.f32"})
	if !slices.Equal(got, []string{"--moe-selection-bias", "/m/bias.f32"}) {
		t.Fatalf("selection bias: %v", got)
	}
	got = appendClassifierArgs(nil, LlamaServerConfig{MoEResident: "/m/res.f32", MoETailSkip: "7"})
	if !slices.Equal(got, []string{"--moe-resident", "/m/res.f32", "--moe-tail-skip", "7"}) {
		t.Fatalf("tail skip: %v", got)
	}
	got = appendClassifierArgs(nil, LlamaServerConfig{ResidCut: "/m/cut.gguf"})
	if !slices.Equal(got, []string{"--resid-cut", "/m/cut.gguf"}) {
		t.Fatalf("resid cut: %v", got)
	}
	// tail skip without a resident file, or with a non-numeric rank, is left out
	for _, c := range []LlamaServerConfig{{MoETailSkip: "7"}, {MoEResident: "/m/res.f32", MoETailSkip: "x"}} {
		if got := appendClassifierArgs(nil, c); len(got) != 0 {
			t.Fatalf("expected no args for %+v, got %v", c, got)
		}
	}
}
