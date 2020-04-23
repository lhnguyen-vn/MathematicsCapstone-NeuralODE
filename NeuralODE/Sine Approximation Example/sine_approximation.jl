### ============================================================================
### Load Packages
### ============================================================================

include("./NeuralODE.jl")
using Plots

### ============================================================================
### Generate Training Data
### ============================================================================

# Random input from -5 to 5
X = rand(Float32, 300) * 10 .- 5 |> sort!

# Sine function output with added noise
Y = sin.(X) .+ Float32(1e-3) * (rand(Float32, 300) .- 0.5f0)

# Visualization of sine function and training data
xgrid = -5:0.1:5
function plot_data()
    scatter(X, Y, label=:none, ms=3, alpha=0.5)    # training data
    plot!(xgrid, sin.(xgrid), label="sine", lw=2, c=:red)    # sine function
end
pl1 = plot_data()

### ============================================================================
### Neural Network Model
### ============================================================================

# Define neural network model
nn_model = Chain(Dense(1, 8, tanh), Dense(8, 1))

# Parameters to be optimized
nn_params = Flux.params(nn_model)

# Sum of squared error as loss function
nn_loss() = sum(abs2, nn_model(reshape(X, 1, :)) - Y')

# Set up to run for 100 epochs
nn_data = Iterators.repeated((), 100)

# Optimizer
nn_opt = ADAM(0.1)

# Store loss each epoch for visualization
nn_losses = Float32[]
nn_cb = () -> begin
    push!(nn_losses, nn_loss())
end

# Training loop
Flux.train!(nn_loss, nn_params, nn_data, nn_opt, cb=nn_cb)

# Plot losses versus epoch
pl2 = plot(1:100, nn_losses, label="Neural Network Model Loss",
     xlabel="Epoch", ylabel="Loss", c=:orange)

# Network Visualization
plot!(pl1, xgrid, nn_model(xgrid')', c=:orange, lw=2, 
      xlabel="x", ylabel="y", label="Neural Network Model")

### ============================================================================
### Naive Neural ODE Model
### ============================================================================

# Derivative model
model = Chain(Dense(1, 8, tanh), Dense(8, 1))
θ, re = Flux.destructure(model)
dzdt(z, θ, t) = re(θ)(z)

# Define Neural ODE
node_model = NeuralODE(dzdt, θ, [0.0f0, 10.0f0])

# Parameters to be optimized, including time span
Flux.trainable(node::NeuralODE) = (node.θ, node.tspan)
node_params = Flux.params(node_model)

# Sum of squared error as loss function
node_loss() = sum(abs2, node_model(reshape(X, 1, :))[end] - Y')

# Set up to run for 100 epochs
node_data = Iterators.repeated((), 100)

# Optimizer
node_opt = ADAM(0.1)

# Store losses for visualization
node_losses = Float32[]
node_cb = () -> begin
    push!(node_losses, node_loss())
end

# Training loop
Flux.train!(node_loss, node_params, node_data, node_opt, cb=node_cb)

# Plot losses versus epochs
plot!(pl2, 1:100, node_losses, label="Neural ODE Model Loss",
      c=:green3, xlabel="Epoch", ylabel="Loss")

# Neural network visualization
plot!(pl1, xgrid, node_model(xgrid')[end]', lw=2, 
      xlabel="x", ylabel="y", label="Neural ODE Model", c=:green3)

### ============================================================================
### Augmented Neural ODE Model
### ============================================================================

# Derivative model
model = Chain(Dense(4, 8, tanh), Dense(8, 4))
θ, re = Flux.destructure(model)
dzdt(z, θ, t) = re(θ)(z)

# Define Augmented Neural ODE
anode_model = NeuralODE(dzdt, θ, [0.0f0, 10.0f0])

# Parameters to be optimized, including time span
anode_params = Flux.params(anode_model)

# Input augmentation
aug_X = hcat(X, zeros(eltype(X), length(X), 3)) |> transpose

# Sum of squared error as loss function
anode_loss() = sum(abs2, anode_model(aug_X)[end][1, :] - Y)

# Set up to run for 100 epochs
anode_data = Iterators.repeated((), 100)

# Optimizer
anode_opt = ADAM(0.01)

# Store losses for visualization
anode_losses = Float32[]
anode_cb = () -> begin
    push!(anode_losses, anode_loss())
end

# Training loop
Flux.train!(anode_loss, anode_params, anode_data, anode_opt, cb=anode_cb)

# Plot losses versus epochs
plot!(pl1, 1:100, anode_losses, label="Augmented Neural ODE Loss",
      xlabel="Epoch", ylabel="Loss", c=:purple)

# Augmented Neural ODE visualization
aug_xgrid = hcat(xgrid, zeros(eltype(xgrid), length(xgrid), 3)) |> transpose
plot!(pl2, xgrid, anode_model(aug_xgrid)[end][1, :], lw=2, c=:purple,
      xlabel="x", ylabel="y", label="Augmented Neural ODE Model")
      
# Plot all visualizations
plot(pl1, pl2, legendfontsize=6, size=(800, 400))

### ============================================================================
### Mathematical Neural ODE Model
### ============================================================================

# Derivative model for [Y, dY]
θ = zeros(Float32, 2, 2)
dzdt(z, θ, t) = θ * z

# Define ODE
ode_model = NeuralODE(dzdt, θ, [X[1], X[end]])

# Parameters to be optimized
Flux.trainable(ode::NeuralODE) = (ode.θ,)
ode_params = Flux.params(ode_model)

# Set up to run for 200 loops
ode_data = Iterators.repeated((), 200)

# Approximation of first derivative from data set 
dY = [(Y[i+1] - Y[i-1])/(X[i+1] - X[i-1]) for i in 2:length(X)-1]
pushfirst!(dY, (Y[2] - Y[1])/(X[2] - X[1]))
push!(dY, (Y[end] - Y[end-1])/(X[end] - X[end-1]))

# Solve for all solutions at each time step X
predict() = ode_model([Y[1], dY[1]], saveat=X)

# Sum of squared error as loss function
ode_loss() = begin
    predicted = predict()
    sum(abs2, hcat(predicted...) - [Y'; dY'])
end

# Optimizer
ode_opt = ADAM(0.1)

# Store losses for visualization
ode_losses = Float32[]
ode_cb = () -> begin
    push!(ode_losses, ode_loss())
end

# Training loop
Flux.train!(ode_loss, ode_params, ode_data, ode_opt, cb=ode_cb)

# Plot losses versus epochs
pl3 = plot(1:200, ode_losses, label="Neural ODE Model Loss",
           c=:orange, xlabel="Epoch", ylabel="Loss")
           
# Mathematical Neural ODE visualization
pl4 = plot_data()
Ŷ = (predicted = predict(); [p[1] for p in predicted])
plot!(pl4, X, Ŷ, lw=2, c=:orange,
      xlabel="x", ylabel="y", label="Neural ODE Model")
      
# Show plots together
plot(pl4, pl3, size=(800, 400))