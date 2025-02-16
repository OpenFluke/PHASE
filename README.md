# PHASE – Programmable, Hyper-adaptive, Scalable Engine

**PHASE** is an advanced AI neural network framework written in Go. It is designed to build flexible, modular, and customizable neural network architectures that evolve over time. With its transparent, JSON-driven configuration, PHASE enables dynamic mutation, species clustering, and distributed execution—running seamlessly on both browser (via WebAssembly) and desktop environments.

---

## Overview

PHASE provides a platform for constructing neural networks with diverse neuron types and emergent behaviors. Its design emphasizes:

- **Modularity:** Every neuron, connection, and parameter is clearly defined and inspectable.
- **Openness:** JSON-based configuration allows you to adjust the architecture without modifying source code.
- **Reconfigurability:** Easily add, remove, or modify neurons and connections to evolve your network structure.
- **Scalability:** Run on multiple nodes or in-browser without needing model conversion.

---

## Core Components

### 1. Benchmarking

PHASE includes an integrated benchmarking suite to measure floating-point operations on both single-threaded and multi-threaded setups. It computes operations per second for both float32 and float64, estimates maximum layers and nodes based on these counts, and formats large numbers into human-readable strings.

*Key Functions:*
- **RunBenchmark:** Runs the benchmark for a specified duration.
- **PerformFloat32Ops / PerformFloat64Ops:** Execute multiply-add operations.
- **EstimateMaxLayersAndNodes:** Provides an estimation of network scalability.

### 2. Activation Functions

The framework supports a variety of scalar activation functions essential for neural computations:

- **ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Linear**

A mapping is maintained so that any neuron can dynamically select its activation function.

### 3. Blueprint

The **Blueprint** is the central structure representing an entire neural network. It holds:

- **Neurons & QuantumNeurons:** Collections of standard and quantum-inspired neurons.
- **InputNodes & OutputNodes:** IDs designating input and output neurons.
- **ScalarActivationMap:** The available activation functions.

It provides methods for initializing the network, performing forward propagation, applying activation functions, and retrieving outputs.

### 4. Neuron Processing

Neurons in PHASE come in various types, including but not limited to:

- **Dense, RNN, LSTM, CNN, BatchNorm, Dropout, Attention, and Neuro-Cellular Automata (NCA) Neurons**

Each type has specialized processing methods to handle its unique computations—from simple dense propagation to convolution and LSTM gating. Additional functionalities include:

- **Dropout and Batch Normalization:** For regularization and stability.
- **Attention Mechanisms:** To dynamically weigh inputs.
- **Convolutional Operations:** For tasks like image processing.

### 5. Mutations & Evolution

PHASE supports dynamic evolution of network architectures via mutation functions that can:

- Add or remove neurons.
- Randomly mutate activation functions, biases, and connection weights.
- Rewire connections between neurons.
- Change neuron types on the fly.

This evolutionary framework enables experimentation with emergent behaviors and species clustering based on blueprint similarity.

### 6. Utilities

A set of utility functions supports tasks like:

- **Loading/Saving JSON Configurations:** Easily serialize or deserialize a network.
- **Downloading Files and Unzipping:** For handling external resources.
- **Softmax Implementation:** For normalizing output neuron values.
- **Miscellaneous Math Helpers:** For operations like element-wise multiplication, summing slices, and safe square-root calculations.

### 7. Species Clustering

PHASE can compute a similarity percentage between blueprints and cluster them into species. This mechanism enables grouping of similar networks based on neuron parameters and connection patterns.

---

## How It Works

1. **Define Your Network:** Use JSON to define neurons, their connections, activation functions, and other configuration details.
2. **Initialize PHASE:** Load the configuration into a new Blueprint instance. The framework initializes activation functions and sets up the network.
3. **Run Inference:** Provide input data and run the forward propagation for a specified number of timesteps.
4. **Benchmark & Evolve:** Use the built-in benchmarking functions to measure performance and apply mutation operators to evolve the network.
5. **Observe Emergent Behavior:** Leverage species clustering and dynamic mutation to explore self-organizing and adaptive patterns.

---

## Getting Started

1. **Setup:** Install Go and initialize your Go module.
2. **Configuration:** Create a JSON file that outlines your neural network’s architecture.
3. **Initialization:** Load your JSON configuration into PHASE to build your Blueprint.
4. **Execution:** Run inference using input data and monitor outputs.
5. **Evolution & Benchmarking:** Use the provided mutation and benchmarking functions to evolve and optimize your network.
6. **Deployment:** For scalable deployment, split your network into shards for distributed processing on clusters or in-browser via WebAssembly.

---

## Future Work

- **Reinforcement Learning:** Integrate reward-driven mechanisms to further evolve network architectures.
- **Training Algorithms:** Implement backpropagation and gradient descent for supervised learning.
- **Model Conversion Tools:** Develop utilities to convert pre-trained models into PHASE’s format.
- **Advanced Visualization:** Build tools for real-time visualization of neuron interactions, mutations, and species clusters.

---


## License

This project is licensed under the [Apache License 2.0](LICENSE).
