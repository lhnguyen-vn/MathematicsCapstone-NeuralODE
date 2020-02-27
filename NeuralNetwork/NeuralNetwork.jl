### Layer struct
struct Layer{T, U, V, F}
    W::T # weight
    b::U # bias
    z::V # intermediate state
    σ::F # activation function
end

# Layer constructor
# Takes in dimensions of the input and the output
# and an activation function
Layer(in::Int, out::Int, σ::Function) =
    Layer(rand(Float32, out, in) .- 0.5f0,
          zeros(Float32, out),
          Array{Float32}[],
          σ)

# Layer evaluation
function (l::Layer)(X)
    W, b, z, σ = l.W, l.b, l.z, l.σ

    # Store intermediate state for back propagation
    empty!(z)
    push!(z, W * X .+ b)

    return σ.(z[end])
end

# Update layer given partial derivatives and learning rate
function update!(l::Layer, dW, db, η)
    l.W .-= η * dW
    l.b .-= η * db
end

# Back propagation given the input from the previous layer and the cost gradient wrt this layer's output
function back_propagate!(l::Layer, ∂Cost∂a_out, a_in, η)
    dσ = derive(l.σ)

    # Cost with respect to z
    ∂Cost∂z = ∂Cost∂a_out .* dσ.(l.z[1])

    # Cost with respect to W
    ∂Cost∂W = sum([∂Cost∂z[:, i] * a_in[:, i]' for i in size(∂Cost∂z, 2)])

    # Cost with respect to b
    ∂Cost∂b = ∂Cost∂z * ones(size(∂Cost∂z, 2))

    # Update parameters
    update!(l, ∂Cost∂W, ∂Cost∂b, η)

    # Cost with respect to input from last layer
    ∂Cost∂a_in = l.W' * ∂Cost∂z
end

### Model struct
struct Model{T<:Tuple, U}
    layers::T
    a::U # layer outputs

    # Model constructor
    Model(layers...) = new{typeof(layers), Vector{Array{Float32}}}(layers, [])
end

# Model evaluation
function (m::Model)(X)
    empty!(m.a)
    push!(m.a, X)

    # Evaluate each layer and store their outputs
    for layer in m.layers
        push!(m.a, layer(m.a[end]))
    end

    # Return final output
    return m.a[end]
end

# Model back propagation
function back_propagate!(m::Model, ∂Cost∂aL, η)
    # Back propagate through each layer
    ∂Cost∂a_out = ∂Cost∂aL
    for layer in reverse(m.layers)
        a_in = pop!(m.a)
        ∂Cost∂a_out = back_propagate!(layer, ∂Cost∂a_out, a_in, η)
    end
end

# Train model with a Cost objective, dataset, and learning rate
function train!(m::Model, Cost, dataset, η)
    costs = Float64[]
    dCost = derive(Cost)

    # Go through batches
    for batch in dataset
        X, Y = batch

        # Forward pass
        output = m(X)

        # Calculate cost
        aL = pop!(m.a)
        cost = Cost(aL, Y)
        push!(costs, cost)

        # Back propagation
        ∂Cost∂aL = dCost(aL, Y)
        back_propagate!(m, ∂Cost∂aL, η)
    end

    return sum(costs) / length(dataset)
end