package phase

import (
	"math"
)

// EvaluateMetrics runs the network on each sample, then calculates:
//   - exactAcc: fraction where argmax(output) == roundedLabel
//   - closeAccs: slice of length 9 for thresholds [0.1 .. 0.9]
//   - proximityScore: measure in [0..100] of how close the correct output is to the top output.
//
// inputs: slice of input maps, each map: inputNeuronID -> floatValue
// labels: slice of float labels, which will be rounded to int
func (bp *Phase) EvaluateMetrics(inputs []map[int]float64, labels []float64) (exactAcc float64, closeAccs []float64, proximityScore float64) {
	nSamples := len(inputs)
	if nSamples == 0 || len(labels) != nSamples {
		// No data or mismatch in lengths
		return 0, nil, 0
	}

	// Basic checks
	numInputs := len(bp.InputNodes)
	numOutputs := len(bp.OutputNodes)
	if numInputs == 0 || numOutputs == 0 {
		return 0, nil, 0
	}

	// We’ll measure “close” at 10%, 20%, ..., 90%.
	thresholds := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	closeCounts := make([]float64, len(thresholds))

	correctExact := 0.0
	totalProx := 0.0
	sampleContribution := 100.0 / float64(nSamples)

	for i, inputMap := range inputs {
		// Round the label
		roundedLabel := int(math.Round(labels[i]))

		// Skip if label is out of range
		if roundedLabel < 0 || roundedLabel >= numOutputs {
			continue
		}
		// Skip if input map has the wrong size
		if len(inputMap) != numInputs {
			continue
		}

		// Run forward pass
		bp.RunNetwork(inputMap, 1)

		// Collect output values
		vals := make([]float64, numOutputs)
		for j := 0; j < numOutputs; j++ {
			v := bp.Neurons[bp.OutputNodes[j]].Value
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			vals[j] = v
		}

		// Identify predicted class
		pred := argmaxFloat64Slice(vals)
		correctVal := vals[roundedLabel]
		maxVal := maxFloat64Slice(vals)
		if maxVal <= 0 || math.IsNaN(maxVal) {
			maxVal = 1.0 // Avoid divide-by-zero
		}

		// Exact match
		if pred == roundedLabel {
			correctExact++
			// If exactly correct, it also counts as “close” for *all* thresholds
			for k := range thresholds {
				closeCounts[k]++
			}
			// For proximity, exactly-correct samples add the full slice contribution
			totalProx += sampleContribution
		} else {
			// Compute how close the correct output is to the top output
			proximityRatio := 0.0
			if correctVal >= 0 {
				if correctVal <= maxVal {
					proximityRatio = correctVal / maxVal
				} else if correctVal <= 2.0*maxVal {
					// If correctVal is up to 2x the top value, do partial ratio
					diff := correctVal - maxVal
					pr := 1.0 - (diff / maxVal)
					if pr < 0 {
						pr = 0
					}
					proximityRatio = pr
				}
			}
			totalProx += proximityRatio * sampleContribution

			// Check “close” at multiple thresholds
			for k, p := range thresholds {
				lowerBound := maxVal * (1.0 - p)
				upperBound := maxVal * (1.0 + p)
				if correctVal >= lowerBound && correctVal <= upperBound {
					closeCounts[k]++
				}
			}
		}
	}

	exactAcc = correctExact / float64(nSamples)
	proximityScore = totalProx // in [0..100]
	closeAccs = make([]float64, len(thresholds))
	for i := range closeAccs {
		closeAccs[i] = closeCounts[i] / float64(nSamples)
	}

	// Sanitize any NaN/Inf
	if math.IsNaN(proximityScore) || math.IsInf(proximityScore, 0) {
		proximityScore = 0
	}
	for i := range closeAccs {
		if math.IsNaN(closeAccs[i]) || math.IsInf(closeAccs[i], 0) {
			closeAccs[i] = 0
		}
	}

	return exactAcc, closeAccs, proximityScore
}
