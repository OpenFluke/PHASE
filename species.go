package blueprint

import (
	"fmt"
	"math"
)

// BlueprintSimilarity computes a similarity percentage (0–100) between two blueprints.
// It compares neurons that have the same ID by looking at the bias and connection weights.
// A value of 100 means the blueprints are identical in the compared parameters.
func BlueprintSimilarity(bp1, bp2 *Blueprint) float64 {
	totalSim := 0.0
	count := 0.0

	// Iterate over all neurons in bp1.
	for id, neuron1 := range bp1.Neurons {
		if neuron2, exists := bp2.Neurons[id]; exists {
			// Compare neuron types. If different, assign a low similarity (or 0).
			typeSim := 1.0
			if neuron1.Type != neuron2.Type {
				typeSim = 0.0 // or you could use a partial penalty like 0.5
			}
			totalSim += typeSim
			count++

			// Compare activation functions (if you want to be sensitive here).
			actSim := 1.0
			if neuron1.Activation != neuron2.Activation {
				actSim = 0.0 // or use a partial penalty
			}
			totalSim += actSim
			count++

			// Compare biases.
			biasDenom := math.Abs(neuron1.Bias) + math.Abs(neuron2.Bias) + 1e-7
			biasSim := 1.0 - math.Abs(neuron1.Bias-neuron2.Bias)/biasDenom
			totalSim += biasSim
			count++

			// Compare connection weights for common connections.
			commonConns := len(neuron1.Connections)
			if len(neuron2.Connections) < commonConns {
				commonConns = len(neuron2.Connections)
			}
			for i := 0; i < commonConns; i++ {
				w1 := neuron1.Connections[i][1]
				w2 := neuron2.Connections[i][1]
				weightDenom := math.Abs(w1) + math.Abs(w2) + 1e-7
				weightSim := 1.0 - math.Abs(w1-w2)/weightDenom
				totalSim += weightSim
				count++
			}
			// Penalize differences in the number of connections.
			diffConns := math.Abs(float64(len(neuron1.Connections) - len(neuron2.Connections)))
			maxConns := math.Max(float64(len(neuron1.Connections)), float64(len(neuron2.Connections)))
			if maxConns > 0 {
				connPenalty := diffConns / maxConns
				// Subtract the penalty from the similarity (or multiply by a factor).
				totalSim += (1.0 - connPenalty)
				count++
			}
		} else {
			// If a neuron is missing in bp2, count it as 0 similarity.
			totalSim += 0.0
			count++
		}
	}

	// Also account for extra neurons in bp2.
	for id := range bp2.Neurons {
		if _, exists := bp1.Neurons[id]; !exists {
			totalSim += 0.0
			count++
		}
	}

	if count == 0 {
		return 0.0
	}

	baseSim := totalSim / count

	// Penalize differences in total neuron count.
	n1 := len(bp1.Neurons)
	n2 := len(bp2.Neurons)
	diff := math.Abs(float64(n1 - n2))
	maxCount := math.Max(float64(n1), float64(n2))
	penalty := diff / maxCount

	finalSim := baseSim * (1.0 - penalty)
	return finalSim * 100.0 // Scale to a percentage (0–100).
}

// ClusterBlueprintsBySpecies groups blueprints into species based on a similarity threshold percentage.
// Two blueprints are considered similar (and thus in the same species) if their similarity is
// greater than or equal to similarityThreshold. The function returns a map where the key is a species ID
// and the value is a slice of blueprint IDs belonging to that species.
func ClusterBlueprintsBySpecies(blueprints map[int]*Blueprint, similarityThreshold float64) map[int][]int {
	// Initialize union-find structure: each blueprint starts in its own set.
	parent := make(map[int]int)
	for id := range blueprints {
		parent[id] = id
	}

	// find returns the representative (root) for a given blueprint ID.
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}

	// union merges the sets for blueprint IDs x and y.
	union := func(x, y int) {
		rootX := find(x)
		rootY := find(y)
		if rootX != rootY {
			parent[rootY] = rootX
		}
	}

	// Get a slice of all blueprint IDs.
	ids := []int{}
	for id := range blueprints {
		ids = append(ids, id)
	}

	// Compare every pair of blueprints.
	for i := 0; i < len(ids); i++ {
		for j := i + 1; j < len(ids); j++ {
			similarity := BlueprintSimilarity(blueprints[ids[i]], blueprints[ids[j]])
			if similarity >= similarityThreshold {
				// If the similarity is above the threshold, merge the two blueprints into the same set.
				union(ids[i], ids[j])
				if blueprints[ids[i]].Debug {
					fmt.Printf("Blueprint %d and Blueprint %d are similar (%.2f%%) and have been clustered together.\n", ids[i], ids[j], similarity)
				}
			}
		}
	}

	// Build clusters from the union-find structure.
	clusters := make(map[int][]int)
	for _, id := range ids {
		root := find(id)
		clusters[root] = append(clusters[root], id)
	}

	return clusters
}
