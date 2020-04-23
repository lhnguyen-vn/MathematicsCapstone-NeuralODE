### ============================================================================
### Load Packages
### ============================================================================

include("./NeuralODE.jl")
using Plots

### ============================================================================
### Define and Solve Lotka-Volterra Model
### ============================================================================

# Lotka-Volterra equations
function model(z, θ, t)
    x, y = z
    α, β, δ, γ = θ
    return [α*x - β*x*y,
            -δ*y + γ*x*y]
end

# Starting populations at time 0
z0 = [1.0f0, 1.0f0]

# Parameters
θ = [2.2f0, 1.0f0, 2.0f0, 0.4f0]

# Define and solve ODE problem
problem = ODEProblem(model, z0, (0.0f0, 10.0f0), θ)
sol = solve(problem, Tsit5())

# Visualize solution
plot(sol)

### ============================================================================
### Neural ODE with Lotka-Volterra equations
### ============================================================================

# Define Neural ODE layer
node = NeuralODE(model, θ, [0.0f0, 10.0f0])

# Parameters to be optimized
params = Flux.params(node)

# Sum of squared error from 1 as loss function
function loss()
    zs = node(z0, saveat=0.1f0)
    return sum(abs2, vcat(zs...) .- 1.0f0)
end

# Set up the Neural ODE 100 to run for 100 epochs
data = Iterators.repeated((), 100)

# Optimizer
opt = ADAM(0.1)

# Update the plot of our populations each epoch
anim = Animation()
cb = () -> begin
    plot(solve(remake(problem; p=node.θ), Tsit5()),
         xlims=(0, 10), ylims=(0, 16))
    frame(anim)
end

# Training Loop
Flux.train!(loss, params, data, opt, cb=cb)

# Training visualization
gif(anim)