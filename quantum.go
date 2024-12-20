package blueprint

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
)

// QuantumState represents a quantum state with amplitude and phase
type QuantumState struct {
	Amplitude complex128
	Phase     float64
}

// QuantumNeuron extends the basic neuron for quantum operations
type QuantumNeuron struct {
	ID            int
	QuantumState  QuantumState
	QuantumGates  []QuantumGate
	Entanglements []EntanglementInfo
	Superposition []complex128
	Connections   [][]complex128 // Quantum weights as complex numbers

	// Additional fields for entanglement and measurement
	EntanglementCreated bool
	IsEntangled         bool
	IsMeasured          bool
}

// QuantumGate represents a quantum operation
type QuantumGate struct {
	Type   string // "Hadamard", "PauliX", "PauliY", "PauliZ", "CNOT"
	Matrix [][]complex128
}

// EntanglementInfo tracks quantum entanglement between neurons
type EntanglementInfo struct {
	PartnerID int
	Type      string // "Bell", "GHZ", "Cluster"
	Strength  float64
}

// ProcessQuantumNeuron handles quantum operations
func (bp *Blueprint) ProcessQuantumNeuron(neuron *QuantumNeuron) {
	// Check if the neuron is entangled
	if len(neuron.Entanglements) > 0 {
		for _, entanglement := range neuron.Entanglements {
			if partner, exists := bp.QuantumNeurons[entanglement.PartnerID]; exists {
				switch entanglement.Type {
				case "Bell":
					// Create Bell state if not already created
					if !neuron.EntanglementCreated && !partner.EntanglementCreated {
						bp.createBellState(neuron, partner)
						neuron.EntanglementCreated = true
						partner.EntanglementCreated = true
					}
					// Apply quantum gates to both entangled qubits
					for _, gate := range neuron.QuantumGates {
						switch gate.Type {
						case "Hadamard":
							neuron.Superposition = applyHadamard(neuron.Superposition)
							partner.Superposition = applyHadamard(partner.Superposition)
							fmt.Printf("After Hadamard gate on Neuron %d and Neuron %d: \n", neuron.ID, partner.ID)
							fmt.Printf("Neuron %d Superposition=%v\n", neuron.ID, neuron.Superposition)
							fmt.Printf("Neuron %d Superposition=%v\n", partner.ID, partner.Superposition)
						case "PauliX":
							neuron.Superposition = applyPauliXToSuperposition(neuron.Superposition)
							partner.Superposition = applyPauliXToSuperposition(partner.Superposition)
							fmt.Printf("After PauliX gate on Neuron %d and Neuron %d: \n", neuron.ID, partner.ID)
							fmt.Printf("Neuron %d Superposition=%v\n", neuron.ID, neuron.Superposition)
							fmt.Printf("Neuron %d Superposition=%v\n", partner.ID, partner.Superposition)
						}
					}
					// Measure entangled qubits
					bp.measureEntangledQubits(neuron, partner)
					return
				}
			}
		}
	}

	// Apply quantum gates to unentangled qubits
	for _, gate := range neuron.QuantumGates {
		switch gate.Type {
		case "Hadamard":
			neuron.Superposition = applyHadamard(neuron.Superposition)
			fmt.Printf("After Hadamard gate on Neuron %d: Superposition=%v\n", neuron.ID, neuron.Superposition)
		case "PauliX":
			neuron.Superposition = applyPauliXToSuperposition(neuron.Superposition)
			fmt.Printf("After PauliX gate on Neuron %d: Superposition=%v\n", neuron.ID, neuron.Superposition)
		case "CNOT":
			bp.applyCNOT(neuron)
		}
	}

	// Normalize the superposition
	neuron.Superposition = normalizeState(neuron.Superposition)

	// Quantum measurement (collapses superposition)
	measuredValue := bp.measureQuantumState(neuron.Superposition)
	neuron.QuantumState.Amplitude = complex(measuredValue, 0)
	fmt.Printf("Quantum Neuron %d measured value: %f\n", neuron.ID, measuredValue)
}

// Helper functions for quantum operations

// normalizeState normalizes the quantum state so that the sum of probabilities equals 1.
func normalizeState(state []complex128) []complex128 {
	var total float64
	for _, amp := range state {
		total += cmplx.Abs(amp) * cmplx.Abs(amp)
	}
	if total == 0 {
		return state
	}
	normalizedState := make([]complex128, len(state))
	sqrtTotal := math.Sqrt(total)
	for i, amp := range state {
		normalizedState[i] = amp / complex(sqrtTotal, 0)
	}
	return normalizedState
}

// applyHadamard applies the Hadamard gate to the superposition.
func applyHadamard(state []complex128) []complex128 {
	h := complex(1/math.Sqrt(2), 0)
	if len(state) == 0 {
		// Initialize to |0⟩ state if superposition is empty
		state = []complex128{1, 0}
	}
	newState := make([]complex128, 2)
	newState[0] = h * (state[0] + state[1])
	newState[1] = h * (state[0] - state[1])
	return normalizeState(newState)
}

// applyPauliXToSuperposition applies the Pauli-X gate to the superposition.
func applyPauliXToSuperposition(state []complex128) []complex128 {
	if len(state) != 2 {
		// Invalid state
		return state
	}
	newState := []complex128{state[1], state[0]}
	return normalizeState(newState)
}

// applyCNOT applies the CNOT gate to entangle qubits.
func (bp *Blueprint) applyCNOT(control *QuantumNeuron) {
	targetID := control.ID + 1 // Adjust as needed
	target, exists := bp.QuantumNeurons[targetID]
	if !exists {
		fmt.Printf("CNOT gate error: Target neuron %d not found.\n", targetID)
		return
	}

	// Apply CNOT operation on the superpositions
	// For simplicity, only handling the basic case where both qubits have 2 basis states
	if len(control.Superposition) != 2 || len(target.Superposition) != 2 {
		fmt.Printf("CNOT operation not supported for current states.\n")
		return
	}

	// Build joint state
	jointState := make([]complex128, 4)
	jointState[0] = control.Superposition[0] * target.Superposition[0] // |00⟩
	jointState[1] = control.Superposition[0] * target.Superposition[1] // |01⟩
	jointState[2] = control.Superposition[1] * target.Superposition[0] // |10⟩
	jointState[3] = control.Superposition[1] * target.Superposition[1] // |11⟩

	// Apply CNOT gate
	// CNOT flips target qubit when control qubit is |1⟩
	// Swap amplitudes of |10⟩ and |11⟩
	jointState[2], jointState[3] = jointState[3], jointState[2]

	// Update individual superpositions
	controlSuperposition := []complex128{
		jointState[0] + jointState[1], // Sum over target qubit
		jointState[2] + jointState[3],
	}
	targetSuperposition := []complex128{
		jointState[0] + jointState[2], // Sum over control qubit
		jointState[1] + jointState[3],
	}

	control.Superposition = normalizeState(controlSuperposition)
	target.Superposition = normalizeState(targetSuperposition)

	fmt.Printf("After CNOT gate:\n")
	fmt.Printf("Control Neuron %d superposition: %v\n", control.ID, control.Superposition)
	fmt.Printf("Target Neuron %d superposition: %v\n", target.ID, target.Superposition)
}

// measureQuantumState collapses the superposition based on quantum measurement postulates.
func (bp *Blueprint) measureQuantumState(superposition []complex128) float64 {
	probabilities := make([]float64, len(superposition))
	for i, amp := range superposition {
		probabilities[i] = cmplx.Abs(amp) * cmplx.Abs(amp)
	}

	fmt.Printf("Measuring quantum state with probabilities: %v\n", probabilities)

	rnd := rand.Float64()
	cumulative := 0.0
	for i, prob := range probabilities {
		cumulative += prob
		if rnd <= cumulative {
			return float64(i)
		}
	}
	return float64(len(superposition) - 1)
}

// measureEntangledQubits simulates the measurement of entangled qubits with correlated outcomes.
func (bp *Blueprint) measureEntangledQubits(q1, q2 *QuantumNeuron) {
	rnd := rand.Float64()
	if rnd < 0.5 {
		// Both qubits collapse to |0⟩
		q1.Superposition = []complex128{1, 0}
		q2.Superposition = []complex128{1, 0}
		q1.QuantumState.Amplitude = 0
		q2.QuantumState.Amplitude = 0
		fmt.Printf("Both qubits collapsed to |0⟩\n")
	} else {
		// Both qubits collapse to |1⟩
		q1.Superposition = []complex128{0, 1}
		q2.Superposition = []complex128{0, 1}
		q1.QuantumState.Amplitude = 1
		q2.QuantumState.Amplitude = 1
		fmt.Printf("Both qubits collapsed to |1⟩\n")
	}
	fmt.Printf("Quantum Neuron %d measured value: %f\n", q1.ID, real(q1.QuantumState.Amplitude))
	fmt.Printf("Quantum Neuron %d measured value: %f\n", q2.ID, real(q2.QuantumState.Amplitude))
}

// createBellState entangles two qubits into a Bell state.
func (bp *Blueprint) createBellState(q1, q2 *QuantumNeuron) {
	// Indicate that the qubits are entangled
	q1.IsEntangled = true
	q2.IsEntangled = true

	fmt.Printf("Created Bell state between Neuron %d and Neuron %d\n", q1.ID, q2.ID)
}

// createGHZState creates a GHZ state among multiple qubits.
func (bp *Blueprint) createGHZState(neurons ...*QuantumNeuron) {
	amplitude := complex(1/math.Sqrt(2), 0)
	for _, neuron := range neurons {
		neuron.Superposition = []complex128{amplitude, 0}
		neuron.IsEntangled = true
	}
	// Note: Proper GHZ state creation would require modeling the joint state of the qubits.
	fmt.Printf("Created GHZ state among neurons: ")
	for _, neuron := range neurons {
		fmt.Printf("%d ", neuron.ID)
	}
	fmt.Printf("\n")
}

// QuantumLayer represents a collection of quantum neurons
type QuantumLayer struct {
	Neurons   []*QuantumNeuron
	Topology  string  // "Sequential", "Entangled", "Hybrid"
	Coherence float64 // Quantum coherence time
}
