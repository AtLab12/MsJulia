using LinearAlgebra

abstract type GraphNode end
abstract type DescentMethod end
abstract type Operator <: GraphNode end


struct Constant{T} <: GraphNode
    output::T
end

using Parameters

@with_kw mutable struct Variable{F} <: GraphNode
    output::F
    gradient::Any = nothing
    name::String = "?"
end

mutable struct ScalarOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs::Any
    output::Any
    gradient::Any
    name::String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

struct GradientDescent <: DescentMethod
    alpha
end

function step!(O::GradientDescent, parameters...)
    alpha = O.alpha
    for parameter in parameters
        parameter.output -= alpha * parameter.gradient
    end
end

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) =
    if isnothing(node.gradient)
        node.gradient = gradient
    else
        node.gradient .+= gradient
    end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    if node isa BroadcastedOperator{typeof(flatten)}
        update!(inputs[1], gradients)
    else
        for (input, gradient) in zip(inputs, gradients)
            update!(input, gradient)
        end
    end
    return nothing
end