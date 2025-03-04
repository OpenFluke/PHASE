package phase

import (
	"math"
)

// EvaluateMetrics does 3 things:
//
// (1) exactAcc in [0..100]%
// (2) closenessBins from 0–10%, 10–20%, …, >90% (each bin is % of total samples)
// (3) approxScore in [0..100], guaranteed >= exactAcc, because correct => full credit.
//
// Strategy for approxScore:
//   - If argmax(predClass) == label, partialCredit = 1.0
//   - Else partialCredit = closeness in [0..1], e.g. 1 - ratio,
//     ratio = difference / expectedVal (or clamp/logic as needed).
//   - sum partialCredit * (100 / nSamples).
//
// If you truly want "expected=1.0" for the correct neuron (MNIST style),
// then difference = |vals[label] - 1.0|. If you have other numeric labels,
// adapt the difference/ratio formula accordingly.
func (bp *Phase) EvaluateMetrics(
	inputs []map[int]float64,
	labels []float64,
) (
	exactAcc float64, // (1) Argmax accuracy in [0..100]%
	closenessBins []float64, // (2) 10 bins from 0–10%, 10–20%, ..., >90%
	approxScore float64, // (3) [0..100], awarding partial credit
) {
	nSamples := len(inputs)
	if nSamples == 0 || len(labels) != nSamples {
		return 0, nil, 0
	}

	numInputs := len(bp.InputNodes)
	numOutputs := len(bp.OutputNodes)
	if numInputs == 0 || numOutputs == 0 {
		return 0, nil, 0
	}

	// Bins: ratio in [0..0.1], [0.1..0.2], …, >0.9 => 10 total
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	binCounts := make([]float64, len(thresholds)+1)

	exactMatches := 0.0
	sumApprox := 0.0
	sampleWeight := 100.0 / float64(nSamples) // each sample's "weight" for approx

	for i, inputMap := range inputs {
		label := int(math.Round(labels[i]))
		if label < 0 || label >= numOutputs {
			continue
		}
		if len(inputMap) != numInputs {
			continue
		}

		// The fix: reset so each sample starts with zeroed neuron states
		//bp.ResetNeuronValues()

		// Forward pass
		bp.RunNetwork(inputMap, 1)

		// Gather outputs
		vals := make([]float64, numOutputs)
		for j := 0; j < numOutputs; j++ {
			v := bp.Neurons[bp.OutputNodes[j]].Value
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			vals[j] = v
		}

		// (1) EXACT ACCURACY
		predClass := argmaxFloatSlice(vals)
		if predClass == label {
			exactMatches++
		}

		// (2) CLOSENESS BINS
		//
		// For MNIST: we typically want the correct neuron's value ~1.0.
		// So difference = |vals[label] - 1.0|.
		// ratio = difference (since expectedVal=1)
		correctVal := vals[label]
		difference := math.Abs(correctVal - 1.0)
		if difference < 0 {
			difference = 0 // not strictly needed, difference is never negative
		}

		// clamp difference if you want, e.g. difference>1 => difference=1
		// so ratio in [0..1]
		if difference > 1 {
			difference = 1
		}
		ratio := difference

		// place ratio into bins
		assigned := false
		for k, th := range thresholds {
			if ratio <= th {
				binCounts[k]++
				assigned = true
				break
			}
		}
		if !assigned {
			// ratio > 0.9
			binCounts[len(thresholds)]++
		}

		// (3) APPROX SCORE using CalculatePercentageMatch
		approx := bp.CalculatePercentageMatch(float64(label), float64(predClass))
		// Convert percentage to fraction (0..1) and apply sample weight
		partialCredit := approx / 100.0
		sumApprox += partialCredit * sampleWeight

	}

	// exactAcc => fraction * 100 => [0..100]
	exactAcc = (exactMatches / float64(nSamples)) * 100.0

	// closenessBins => fraction * 100 => each bin in [0..100]
	closenessBins = make([]float64, len(binCounts))
	for i := range binCounts {
		closenessBins[i] = (binCounts[i] / float64(nSamples)) * 100.0
	}

	// sumApprox is already aggregated => [0..100]
	approxScore = sumApprox

	return exactAcc, closenessBins, approxScore
}

func (bp *Phase) CalculatePercentageMatch(expected, actual float64) float64 {
	// Convert to absolute values since we want closeness regardless of sign
	expected = math.Abs(expected)
	actual = math.Abs(actual)

	// If both are 0, it's a perfect match
	if expected == 0 && actual == 0 {
		return 100.0
	}

	// If one is 0 and other isn't, calculate based on difference
	if expected == 0 || actual == 0 {
		maxValue := math.Max(expected, actual)
		difference := math.Abs(expected - actual)
		// Percentage decreases as difference from 0 increases
		if maxValue == 0 {
			return 100.0 // Safety check
		}
		closeness := 1 - (difference / (maxValue + 1))
		return math.Max(0, closeness*100)
	}

	// Normal case: neither number is 0
	ratio := actual / expected
	if ratio > 1 {
		ratio = 1 / ratio // Invert if actual > expected
	}

	return ratio * 100
}
