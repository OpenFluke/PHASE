package phase

import (
	"fmt"
	"math/rand"
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
