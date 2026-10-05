package cmd

import (
	"errors"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
)

// trainCmd: ollama train BASE NEW -f data.txt trains a LoRA adapter on the loaded base model
// (llama-server --lora-train) and saves the base model plus the adapter as NEW.
func trainCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "train BASE NEW",
		Short:   "Train a LoRA adapter on a model and save it as a new model (server needs OLLAMA_LORA_TRAIN=1)",
		Args:    cobra.ExactArgs(2),
		PreRunE: checkServerHeartbeat,
		RunE:    TrainHandler,
	}
	cmd.Flags().StringP("file", "f", "", "Training text file (required)")
	cmd.Flags().Int("rank", 8, "LoRA rank")
	cmd.Flags().Float32("alpha", 0, "LoRA alpha (default 2*rank)")
	cmd.Flags().Float32("lr", 1e-4, "AdamW learning rate")
	cmd.Flags().Int("epochs", 1, "Epochs over the training text")
	cmd.Flags().Int("num-ctx", 256, "Tokens per training window (a multiple of 256)")
	cmd.Flags().String("targets", "", "Comma-separated weights to adapt (default attention + FFN)")
	cmd.Flags().String("priority", "idle", "idle: train only while the model serves no request; shared: also between requests")
	return cmd
}

func TrainHandler(cmd *cobra.Command, args []string) error {
	file, _ := cmd.Flags().GetString("file")
	if file == "" {
		return errors.New("a training text file is required (-f)")
	}
	text, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	rank, _ := cmd.Flags().GetInt("rank")
	alpha, _ := cmd.Flags().GetFloat32("alpha")
	lr, _ := cmd.Flags().GetFloat32("lr")
	epochs, _ := cmd.Flags().GetInt("epochs")
	numCtx, _ := cmd.Flags().GetInt("num-ctx")
	targets, _ := cmd.Flags().GetString("targets")
	priority, _ := cmd.Flags().GetString("priority")

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	var bar *progress.Bar
	var spinner *progress.Spinner
	status := ""
	fn := func(resp api.ProgressResponse) error {
		if resp.Total > 0 {
			if bar == nil {
				if spinner != nil {
					spinner.Stop()
				}
				bar = progress.NewBar("training", resp.Total, resp.Completed)
				p.Add("training", bar)
			}
			bar.Set(resp.Completed)
			return nil
		}
		if resp.Status != status {
			if spinner != nil {
				spinner.Stop()
			}
			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}
		return nil
	}

	req := &api.TrainRequest{
		Model:        args[0],
		Name:         args[1],
		Text:         string(text),
		Rank:         rank,
		Alpha:        alpha,
		LearningRate: lr,
		Epochs:       epochs,
		NumCtx:       numCtx,
		Targets:      targets,
		Priority:     priority,
	}
	if err := client.Train(cmd.Context(), req, fn); err != nil {
		return err
	}
	p.Stop()
	fmt.Fprintf(os.Stderr, "created %s (%s + trained LoRA adapter)\n", args[1], args[0])
	return nil
}
