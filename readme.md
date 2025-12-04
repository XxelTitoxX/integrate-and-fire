# Integrate-and-Fire Neuron Simulation

This project provides a Python implementation of a stochastic Integrate-and-Fire
neuron model driven by a Poisson random measure. The membrane potential evolves
according to a leaky drift ODE and undergoes state-dependent random spikes
generated using a Poisson thinning algorithm.

Between spikes, the potential is updated analytically using the exact solution
of the leaky ODE, ensuring accuracy without numerical integration errors.

A plotting script is included to visualize the membrane potential trajectory and
the spike times.

## Features

- Leaky Integrate-and-Fire drift with exact analytical solution  
- Poisson thinning for random spike generation  
- State-dependent firing rate  
- Spike resets  
- Visualization of the simulated membrane potential  

## Requirements

- Python >= 3.9  
- NumPy  
- Matplotlib  

Install dependencies with:
    ```bash
    pip install numpy matplotlib
    ```

## How to Run

Run the simulation script:
    ```bash
    python simulation.py
    ```

This will generate a plot showing the membrane potential over time and the spike
events.
