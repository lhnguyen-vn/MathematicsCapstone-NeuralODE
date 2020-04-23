### Author: Long Nguyen


### ============================================================================
### Layer Struct
### ============================================================================

# A Layer consists of weights, bias, and an activation function to produce the
# output given the input. Additionally, we also save the result before applying 
# the activation function to compute the backward pass.
struct Layer{WS, BS, Z, F}
    W::WS    # weights
    b::BS    # biases
    z::Z     # intermediate state
    σ::F     # activation function
end

# The Layer constructor takes in the dimensions of the input, output, and an 
# activation function. Weight and biases are set accordingly, and an empty array
# is set up to store the intermediate state.
Layer(in::Int, out::Int, σ::Function) =
    Layer(rand(Float32, out, in) .- 0.5f0,    # weights
          zeros(Float32, out),                # biases
          Array{Float32}[],                   # intermediate state vector
          σ)                                  # activation function

# Layer output is computed using the formula σ(W * X + b)
function (l::Layer)(X)
    W, b, z, σ = l.W, l.b, l.z, l.σ
    temp = W * X .+ b    # .+ broadcasting is also compatible with batches
    empty!(z) 
    push!(z, temp)    # store intermediate state for back propagation
    return σ.(temp)    # apply the activation function element-wise
end

# Layer is updated with partial derivatives and learning rate.
function update!(l::Layer, dW, db, η)
    l.W .-= η * dW
    l.b .-= η * db
end

# Given the derivative of the Cost wrt the ouput, we calculate the partial 
# derivatives with respect to weights, biases, and the input.
function derive(l::Layer, ∂Cost∂a_out, a_in)
    dσ = derive(l.σ)    # user-defined derivative function of σ
    
    # Cost wrt intermediate result
    ∂Cost∂z = ∂Cost∂a_out .* dσ.(l.z[1]) 
    
    # ∂W computes Cost wrt weights for one pair of input and output
    ∂W(∂Cost∂z, a_in) = ∂Cost∂z * a_in'
    # Cost wrt weights for entire batch
    ∂Cost∂W = sum(∂W.(eachcol(∂Cost∂z), eachcol(a_in)))

    # Cost wrt to bias
    ∂Cost∂b = sum(eachcol(∂Cost∂z))

    # Cost wrt input from last layer
    ∂Cost∂a_in = l.W' * ∂Cost∂z

    return ∂Cost∂W, ∂Cost∂b, ∂Cost∂a_in
end

# Back propagation given the input from the previous layer
# and the cost gradient wrt this layer's output
function back_propagate!(l::Layer, ∂Cost∂a_out, a_in, η)
    ∂Cost∂W, ∂Cost∂b, ∂Cost∂a_in = derive(l, ∂Cost∂a_out, a_in)    # gradients
    update!(l, ∂Cost∂W, ∂Cost∂b, η)    # update parameters
    return ∂Cost∂a_in    # Cost wrt input from last layer
end

### ============================================================================
### Model Struct
### ============================================================================

# A Model consists of multiple Layers. Additionally, we store each Layer's
# outputs in an array for the backward pass.
struct Model{LS, OS}
    layers::LS    # Layers
    a::OS         # Layer outputs

    # Model Constructor
    Model(layers...) = new{typeof(layers), Vector{Array{Float32}}}(layers, [])
end

# Evaluate Model by evaluating the layers sequentially.
function (m::Model)(X)
    # Store Model input
    empty!(m.a)
    push!(m.a, X)

    # Evaluate each layer and store their outputs
    for layer in m.layers
        push!(m.a, layer(m.a[end]))
    end

    # Return Model output
    return pop!(m.a)
end

# Back-propagate through the Model by back-propagating through each layer.
function back_propagate!(m::Model, ∂Cost∂aL, η)
    # Back propagate through each layer
    ∂Cost∂a_out = ∂Cost∂aL
    for layer in reverse(m.layers)
        a_in = pop!(m.a)    # retrieve layer input
        ∂Cost∂a_out = back_propagate!(layer, ∂Cost∂a_out, a_in, η)
    end
end

# Training requires a Model, a Cost function, the training dataset, and the 
# learning rate. 
function train!(m::Model, Cost, dataset, η)
    costs = Float32[]    # store cost of each batch in dataset
    dCost = derive(Cost)    # user-defined derivative function of Cost

    # Train Model on each batch in dataset
    for batch in dataset
        X, Y = batch
        out = m(X)

        # Calculate cost
        cost = Cost(out, Y)
        push!(costs, cost)

        # Back propagation
        ∂Cost∂out = dCost(out, Y)
        back_propagate!(m, ∂Cost∂out, η)
    end

    # Return average cost of all batches
    return sum(costs) / length(dataset)
end