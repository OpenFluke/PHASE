package phase

import (
	"fmt"
	"math"
)

// ManualClampNeuronValues iterates over all neurons and clamps each neuron’s
// current Value and (for LSTM) CellState into [minVal, maxVal].
func (bp *Phase) ManualClampNeuronValues(minVal, maxVal float64) {
	if minVal > maxVal {
		// Swap them if user provided inverted range
		fmt.Printf("Warning: minVal > maxVal. Swapping them.\n")
		minVal, maxVal = maxVal, minVal
	}
	for _, neuron := range bp.Neurons {
		// Clamp main neuron value
		if math.IsNaN(neuron.Value) || math.IsInf(neuron.Value, 0) {
			neuron.Value = 0
		} else if neuron.Value > maxVal {
			neuron.Value = maxVal
		} else if neuron.Value < minVal {
			neuron.Value = minVal
		}

		// If LSTM, clamp the cell state
		if math.IsNaN(neuron.CellState) || math.IsInf(neuron.CellState, 0) {
			neuron.CellState = 0
		} else if neuron.CellState > maxVal {
			neuron.CellState = maxVal
		} else if neuron.CellState < minVal {
			neuron.CellState = minVal
		}
	}
}

// ManualClampBiases clamps the Bias field of all neurons into [minVal, maxVal].
func (bp *Phase) ManualClampBiases(minVal, maxVal float64) {
	if minVal > maxVal {
		fmt.Printf("Warning: minVal > maxVal. Swapping them.\n")
		minVal, maxVal = maxVal, minVal
	}
	for _, neuron := range bp.Neurons {
		if math.IsNaN(neuron.Bias) || math.IsInf(neuron.Bias, 0) {
			neuron.Bias = 0
		} else if neuron.Bias > maxVal {
			neuron.Bias = maxVal
		} else if neuron.Bias < minVal {
			neuron.Bias = minVal
		}
	}
}

// ManualClampWeights clamps each connection weight into [minVal, maxVal].
func (bp *Phase) ManualClampWeights(minVal, maxVal float64) {
	if minVal > maxVal {
		fmt.Printf("Warning: minVal > maxVal. Swapping them.\n")
		minVal, maxVal = maxVal, minVal
	}
	for _, neuron := range bp.Neurons {
		for i := range neuron.Connections {
			w := neuron.Connections[i][1]
			if math.IsNaN(w) || math.IsInf(w, 0) {
				w = 0
			} else if w > maxVal {
				w = maxVal
			} else if w < minVal {
				w = minVal
			}
			neuron.Connections[i][1] = w
		}
	}
}
