package phase

import "math/rand"

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

	// Normalize improvements
	normDeltaExactAcc := deltaExactAcc / 100.0
	normDeltaApproxScore := deltaApproxScore / 100.0
	normDeltaClosenessQuality := deltaClosenessQuality / 100.0

	// Weighted sum
	weightExactAcc := 0.3
	weightCloseness := 0.4
	weightApproxScore := 0.3
	return (weightExactAcc * normDeltaExactAcc) +
		(weightCloseness * normDeltaClosenessQuality) +
		(weightApproxScore * normDeltaApproxScore)
}

// **tournamentSelection** selects the best model from a random subset of results.
func (bp *Phase) TournamentSelection(results []ModelResult, currentExactAcc, currentClosenessQuality, currentApproxScore float64, tournamentSize int) ModelResult {
	if len(results) < tournamentSize {
		tournamentSize = len(results)
	}
	perm := rand.Perm(len(results))
	selectedIndices := perm[:tournamentSize]
	bestIdx := selectedIndices[0]
	bestImprovement := bp.ComputeTotalImprovement(results[bestIdx], currentExactAcc, currentClosenessQuality, currentApproxScore)
	for _, idx := range selectedIndices[1:] {
		improvement := bp.ComputeTotalImprovement(results[idx], currentExactAcc, currentClosenessQuality, currentApproxScore)
		if improvement > bestImprovement {
			bestImprovement = improvement
			bestIdx = idx
		}
	}
	return results[bestIdx]
}

func (bp *Phase) ComputeClosenessQuality(bins []float64) float64 {
	quality := 0.0
	for i := 5; i < len(bins); i++ {
		quality += bins[i]
	}
	return quality
}
