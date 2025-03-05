package phase

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// SaveCheckpointsToDirectory saves each checkpoint to a separate file in the specified directory.
func (bp *Phase) SaveCheckpointsToDirectory(inputs []map[int]float64, timesteps int, dirPath string) error {
	// Create or clear the directory
	if err := os.RemoveAll(dirPath); err != nil {
		return fmt.Errorf("failed to clear checkpoint directory %s: %v", dirPath, err)
	}
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory %s: %v", dirPath, err)
	}

	// Save each checkpoint
	for i, inputMap := range inputs {
		bp.Forward(inputMap, timesteps)
		checkpoint := make(map[int]map[string]interface{})
		for id, neuron := range bp.Neurons {
			if neuron.Type != "input" {
				checkpoint[id] = bp.GetNeuronState(neuron)
			}
		}
		filePath := filepath.Join(dirPath, fmt.Sprintf("checkpoint_sample_%d.json", i))
		data, err := json.Marshal(checkpoint)
		if err != nil {
			return fmt.Errorf("failed to marshal checkpoint %d: %v", i, err)
		}
		if err := os.WriteFile(filePath, data, 0644); err != nil {
			return fmt.Errorf("failed to write checkpoint %d to %s: %v", i, filePath, err)
		}
	}
	return nil
}

// EvaluateMetricsFromCheckpointDir evaluates metrics by loading checkpoints from files in batches.
func (bp *Phase) EvaluateMetricsFromCheckpointDir(dirPath string, labels []float64, batchSize int) (float64, []float64, float64) {
	files, err := filepath.Glob(filepath.Join(dirPath, "checkpoint_sample_*.json"))
	if err != nil {
		log.Fatalf("Failed to glob checkpoint files: %v", err)
	}
	sort.Strings(files) // Ensure consistent order

	nSamples := len(files)
	if nSamples != len(labels) {
		log.Fatalf("Mismatch between checkpoint files (%d) and labels (%d)", nSamples, len(labels))
	}

	var exactMatches float64
	binCounts := make([]float64, 10) // 10 bins: 0-10%, 10-20%, ..., >90%
	sumApprox := 0.0

	// Process in batches
	for start := 0; start < nSamples; start += batchSize {
		end := start + batchSize
		if end > nSamples {
			end = nSamples
		}
		batchFiles := files[start:end]
		batchLabels := labels[start:end]

		// Load and process batch
		for i, file := range batchFiles {
			data, err := os.ReadFile(file)
			if err != nil {
				log.Fatalf("Failed to read checkpoint file %s: %v", file, err)
			}
			var checkpoint map[int]map[string]interface{}
			if err := json.Unmarshal(data, &checkpoint); err != nil {
				log.Fatalf("Failed to unmarshal checkpoint from %s: %v", file, err)
			}

			outputs := bp.ComputeOutputsWithNewNeuronsFromCheckpoint(checkpoint)
			vals := make([]float64, len(bp.OutputNodes))
			for j, id := range bp.OutputNodes {
				v := outputs[id]
				if math.IsNaN(v) || math.IsInf(v, 0) {
					v = 0
				}
				vals[j] = v
			}

			label := int(math.Round(batchLabels[i]))
			predClass := argmaxFloatSlice(vals)
			if predClass == label {
				exactMatches++
			}

			correctVal := vals[label]
			difference := math.Abs(correctVal - 1.0)
			if difference > 1 {
				difference = 1
			}
			ratio := difference

			assigned := false
			for k := 0; k < 9; k++ {
				if ratio <= float64(k+1)/10.0 {
					binCounts[k]++
					assigned = true
					break
				}
			}
			if !assigned {
				binCounts[9]++
			}

			approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
			sumApprox += approx / 100.0
		}
	}

	exactAcc := (exactMatches / float64(nSamples)) * 100.0
	closenessBins := make([]float64, 10)
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}
	approxScore := (sumApprox / float64(nSamples)) * 100.0

	return exactAcc, closenessBins, approxScore
}
