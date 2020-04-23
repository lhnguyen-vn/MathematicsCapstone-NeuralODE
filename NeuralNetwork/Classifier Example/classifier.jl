### ============================================================================
### Load Packages
### ============================================================================

using Plots, LinearAlgebra
import Random
include("./NeuralNetwork.jl")

### ============================================================================
### Generate Training Data
### ============================================================================

# True model to classify data
model(x) = x^3 - 3x + 2

# Generate a data set given the classification model, the range of coordinates,
# and the size of the data set.
function generate_dataset(model, xrange, yrange, size)
    # Random sampling function
    sample(range, size) = rand(size) * (range[end] - range[1]) .+ range[1]
    
    # Sample x and y coordinates
    xs = sample(xrange, size)
    ys = sample(yrange, size)

    # Input matrix
    input = hcat(x1s, x2s)'

    # Points above and below the model are labeled 1 and 0, respectively
    output = Float32.(x2s .≥ model.(x1s))'

    return input, output
end

# Coordinate range of data set
x1grid = -3:0.1:3
x2grid = -16:0.1:20

# Generate the inputs and corresponding outputs
input, output = generate_dataset(model, x1grid, x2grid, 500)

# Plot the data set
ishigher = vec(output .== 1)
function plot_data()
    scatter(input[1, ishigher], input[2, ishigher], label="", color=:blue)
    scatter!(input[1, .!ishigher], input[2, .!ishigher], label="", color=:red)
    plot!(x1grid, model.(x1grid), label="True Model", legend=:outertopright)
end
plot_data()

### ============================================================================
### Training Environment
### ============================================================================

### Activation functions
### ====================
# Sigmoid activation function and its derivative
sigmoid(x) = 1 / (1 + exp(-x))
derive(::typeof(sigmoid)) = x -> sigmoid(x) * (1 - sigmoid(x))

# Relu activation function and its derivative
relu(x) = max(0, x)
derive(::typeof(relu)) = x -> (x ≥ 0 ? 1 : 0)

### Neural network model
### ====================
neural_model = Model(
    Layer(2, 10, relu),
    Layer(10, 1, sigmoid))

### Cost objective
### ==============
# Binary cross entropy loss and its derivative
bcentropy(ŷ, y) = -y* log(ŷ + 1e-7) - (1 - y) * log(1 - ŷ + 1e-7)
derive(::typeof(bcentropy)) = (ŷ, y) -> -y/(ŷ + 2e-7) + (1 - y)/(1 - ŷ + 1e-7)

# Cost function and its derivative
Cost(Ŷ, Y) = sum(bcentropy.(Ŷ, Y)) / size(Y, 2)
derive(::typeof(Cost)) = (Ŷ, Y) -> derive(bcentropy).(Ŷ, Y) / size(Y, 2)

### Minibatching
### ============
function minibatch(input, output, batch_size)
    inputs = Array[]
    outputs = Array[]

    # Shuffle data set and split into batches
    rand_idxs = Random.randperm(size(input, 2))
    batch_idxs = Iterators.partition(rand_idxs, batch_size)
    for batch_idx in batch_idxs
        push!(inputs, input[:, batch_idx])
        push!(outputs, output[:, batch_idx])
    end

    # Return data set
    zip(inputs, outputs) |> collect
end

### Model visualization
### ===================
function plot_model(neural_model, x1grid, x2grid)
    plot_data()    # plot true model

    # Find the classification boundary 
    bound = fill(maximum(x2grid), size(x1grid))
    for (i, x1) in enumerate(x1grid)
        for x2 in x2grid
            out = neural_model([x1, x2])[end]
            if out ≥ 0.5
                bound[i] = x2
                break
            end
         end
    end
    
    # Plot neural network model
    plot!(x1grid, bound, label="Neural Network Model")
end

### ============================================================================
### Training
### ============================================================================

η = 0.05    # learning rate
epochs = 1000    # training epochs
anim = Animation()    # visualization of training process

### Training loop
### =============
for i in 1:epochs
    # Prepare batches
    dataset = minibatch(input, output, 5)

    # Train model
    cost = train!(neural_model, Cost, dataset, η)

    # Report loss
    println("Epoch $i average cost: $cost")

    # Update visualization
    if i % 10 == 0
        plot_model(neural_model, x1grid, x2grid)
        frame(anim)
    end
end

### Trained model visualization
### ===========================
gif(anim)
plot_model(neural_model, x1grid, x2grid)