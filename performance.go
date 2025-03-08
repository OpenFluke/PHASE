package phase

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
)

type ModelResult struct {
	BP            *Phase
	ExactAcc      float64   // Exact accuracy in [0, 100]
	ClosenessBins []float64 // Closeness bins in [0, 100] per bin
	ApproxScore   float64   // Approx score in [0, 100]
	NeuronsAdded  int
}

// **computeTotalImprovement** calculates the weighted sum of improvements for a model.
func (bp *Phase) ComputeTotalImprovement(result ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64) float64 {
	newClosenessQuality := bp.ComputeClosenessQuality(result.ClosenessBins)
	deltaExactAcc := result.ExactAcc - currentExactAcc
	deltaApproxScore := result.ApproxScore - currentApproxScore
	deltaClosenessQuality := newClosenessQuality - currentClosenessQuality
	normDeltaExactAcc := deltaExactAcc / 100.0
	normDeltaApproxScore := deltaApproxScore / 100.0
	normDeltaClosenessQuality := deltaClosenessQuality / 100.0
	weightExactAcc := 0.2
	weightCloseness := 0.3
	weightApproxScore := 0.5
	return (weightExactAcc * normDeltaExactAcc) +
		(weightCloseness * normDeltaClosenessQuality) +
		(weightApproxScore * normDeltaApproxScore)
}

// printModelDetails displays detailed metrics for a model during tournament selection.
func (bp *Phase) printModelDetails(result ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64) {
	// Compute Closeness Quality from the model's bins
	newClosenessQuality := bp.ComputeClosenessQuality(result.ClosenessBins)
	// Calculate the differences (deltas) from current metrics
	deltaExactAcc := result.ExactAcc - currentExactAcc
	deltaClosenessQuality := newClosenessQuality - currentClosenessQuality
	deltaApproxScore := result.ApproxScore - currentApproxScore
	// Compute the total improvement score
	improvement := bp.ComputeTotalImprovement(result, currentExactAcc, currentClosenessQuality, currentApproxScore)
	// Print all metrics with deltas and improvement
	fmt.Printf("Model %d: ExactAcc=%.4f (Δ %.4f), ClosenessQuality=%.4f (Δ %.4f), ApproxScore=%.4f (Δ %.4f), Improvement=%.4f\n",
		result.BP.ID, result.ExactAcc, deltaExactAcc, newClosenessQuality, deltaClosenessQuality, result.ApproxScore, deltaApproxScore, improvement)
}

// TournamentSelection selects the best model from a random subset and logs details if debug is enabled.
func (bp *Phase) TournamentSelection(results []ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64, tournamentSize int) ModelResult {
	// Adjust tournament size if there are fewer results
	if len(results) < tournamentSize {
		tournamentSize = len(results)
	}
	// Randomly select indices for the tournament
	perm := rand.Perm(len(results))
	selectedIndices := perm[:tournamentSize]

	// Start with the first model as the best
	bestIdx := selectedIndices[0]
	bestImprovement := bp.ComputeTotalImprovement(results[bestIdx], currentExactAcc, currentClosenessQuality, currentApproxScore)

	// If debug is enabled, print the header and first model's details
	if bp.Debug {
		fmt.Println("Tournament Selection Details:")
		bp.printModelDetails(results[bestIdx], currentExactAcc, currentClosenessQuality, currentApproxScore)
	}

	// Evaluate the remaining models in the subset
	for _, idx := range selectedIndices[1:] {
		improvement := bp.ComputeTotalImprovement(results[idx], currentExactAcc, currentClosenessQuality, currentApproxScore)
		// Log details if debug is on
		if bp.Debug {
			bp.printModelDetails(results[idx], currentExactAcc, currentClosenessQuality, currentApproxScore)
		}
		// Update the best model if this one has higher improvement
		if improvement > bestImprovement {
			bestImprovement = improvement
			bestIdx = idx
		}
	}

	// If debug is enabled, show which model was selected
	if bp.Debug {
		fmt.Printf("Selected Model %d with Improvement: %.4f\n", results[bestIdx].BP.ID, bestImprovement)
	}

	// Return the best model
	return results[bestIdx]
}

func (bp *Phase) ComputeClosenessQuality(bins []float64) float64 {
	quality := 0.0
	weights := []float64{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1} // Higher weight for lower bins
	for i := 0; i < len(bins)-1; i++ {                                     // Exclude >0.9 bin if desired
		quality += bins[i] * weights[i]
	}
	return quality
}

func (bp *Phase) SelectBestModel(results []ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64) (ModelResult, float64) {
	var bestModel ModelResult
	bestImprovement := -math.MaxFloat64 // Start with the lowest possible value

	for _, model := range results {
		improvement := bp.ComputeTotalImprovement(model, currentExactAcc, currentClosenessQuality, currentApproxScore)
		if improvement > bestImprovement {
			bestImprovement = improvement
			bestModel = model
		}
	}

	return bestModel, bestImprovement
}

// EvaluateAndExportToCSV evaluates the model against samples and exports inputs, expected outputs, and current outputs to a CSV file.
// The filename is based on the generation number to avoid overwriting.
func (bp *Phase) EvaluateAndExportToCSV(samples *[]Sample, filePath string, timesteps int) error {
	if bp == nil {
		return fmt.Errorf("model is nil")
	}
	if samples == nil || len(*samples) == 0 {
		return fmt.Errorf("samples are empty or nil")
	}

	// Create the CSV file with the specified path (won't overwrite due to generation-specific naming)
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create CSV file %s: %v", filePath, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Get output node IDs for consistent ordering
	outputNodes := bp.GetOutputIds()
	if len(outputNodes) == 0 {
		return fmt.Errorf("no output nodes defined in the model")
	}

	// Prepare header
	header := []string{"SampleIndex"}
	// Add input keys (assuming 784 inputs for MNIST, indexed 0 to 783)
	for i := 0; i < 784; i++ {
		header = append(header, fmt.Sprintf("Input%d:%d", i, i))
	}
	// Add expected output keys
	for _, nodeID := range outputNodes {
		header = append(header, fmt.Sprintf("ExpectedOutput%d:%d", nodeID, nodeID))
	}
	// Add current output keys
	for _, nodeID := range outputNodes {
		header = append(header, fmt.Sprintf("CurrentOutput%d:%d", nodeID, nodeID))
	}

	// Write header
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %v", err)
	}

	// Process each sample
	for i, sample := range *samples {
		// Reset neuron values to avoid carryover
		bp.ResetNeuronValues()

		// Perform forward pass with the sample's inputs
		bp.Forward(sample.Inputs, timesteps)

		// Get current outputs
		currentOutputs := bp.GetOutputs()

		// Prepare row data
		row := []string{fmt.Sprintf("%d", i)}

		// Add input values
		for j := 0; j < 784; j++ {
			if value, exists := sample.Inputs[j]; exists {
				row = append(row, fmt.Sprintf("%.6f", value))
			} else {
				row = append(row, "0.000000") // Default to 0 if input not present
			}
		}

		// Add expected outputs (sorted by node ID for consistency)
		sortedOutputNodes := make([]int, len(outputNodes))
		copy(sortedOutputNodes, outputNodes)
		sort.Ints(sortedOutputNodes)
		for _, nodeID := range sortedOutputNodes {
			if value, exists := sample.ExpectedOutputs[nodeID]; exists {
				row = append(row, fmt.Sprintf("%.6f", value))
			} else {
				row = append(row, "0.000000") // Default to 0 if not specified
			}
		}

		// Add current outputs
		for _, nodeID := range sortedOutputNodes {
			if value, exists := currentOutputs[nodeID]; exists {
				row = append(row, fmt.Sprintf("%.6f", value))
			} else {
				row = append(row, "0.000000") // Default to 0 if not computed
			}
		}

		// Write row to CSV
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write row %d to CSV: %v", i, err)
		}
	}

	fmt.Printf("Evaluation exported to CSV file: %s\n", filePath)
	return nil
}
