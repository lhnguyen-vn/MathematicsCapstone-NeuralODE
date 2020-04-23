# Mathematics Capstone: Neural ODE
This repository holds the code used for our [Mathematics Capstone Project](https://scholarworks.arcadia.edu/showcase/2020/comp_sci_math/8/) on neural networks and neural ordinary differential equations (ODEs). 

## Implementations
With the goal to fully understand the algorithms and to feature the strengths of the Julia language, we build our own implementation framework of neural networks and neural ODEs using as few external packages as possible. 

- [Neural Network Source Code](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralNetwork/NeuralNetwork.jl)
- [Neural ODE Source Code](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralODE/NeuralODE.jl)

For neural ODEs, we rely on Julia's packages [DifferentialEquations](https://github.com/SciML/DifferentialEquations.jl), [Flux](https://github.com/FluxML/Flux.jl), and [Zygote](https://github.com/FluxML/Zygote.jl). Dependencies can be found in our [Project.toml](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralODE/Project.toml) and [Manifest.toml](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralODE/Manifest.toml) files.

## Experiments
We conduct a few simple experiments as proofs of concept of our implementations. The Lotka-Volterra example is adapted from Julia's [DiffEqFlux](https://github.com/SciML/DiffEqFlux.jl) package for neural ODEs.

- [Neural Network Classifier Example](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralNetwork/Classifer Example)
- [Lotka-Volterra Example](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralODE/Lotka-Volterra Example)
- [Sine Approximation Comparisons](https://lhnguyen-vn.github.io/MathematicsCapstone-NeuralODE/NeuralODE/Sine Approximation Example)
