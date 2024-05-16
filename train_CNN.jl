using MLDatasets
import Flux: onehotbatch
import LinearAlgebra: mul!
include("operators.jl")
include("structures_graph_pass.jl")


train_data_x, train_data_y = MNIST(split=:train)[:]
test_data_x, test_data_y = MNIST(split=:test)[:]

train_data_x = reshape(train_data_x, size(train_data_x, 1), size(train_data_x, 2), 1, size(train_data_x, 3))
test_data_x = reshape(test_data_x, size(test_data_x, 1), size(test_data_x, 2), 1, size(test_data_x, 3))

train_y_outputs = train_data_y
test_y_outputs = test_data_y

train_data_y = convert(Matrix{Float64}, onehotbatch(train_data_y, sort(unique(train_data_y)))')
test_data_y = convert(Matrix{Float64}, onehotbatch(test_data_y, sort(unique(test_data_y)))')

train_size = size(train_data_x, 4)
test_size = size(test_data_x, 4)

input_x = Variable{Any}(output=zeros(4))
input_y = Variable{Any}(output=zeros(10))

settings = (
    epochs=3,
    eta=0.01,
    batch_size=100
)

optimize = GradientDescent(settings.eta)
kernel_size = 3
conv_input = 1
conv_output = 6
flatten_neurons = 1014
hidden_neurons = 84
output_neurons = 10

wc = Variable{Array{Float64,4}}(output=(randn(kernel_size, kernel_size, conv_input, conv_output) .* 0.1))
wh = Variable{Matrix{Float64}}(output=(randn(hidden_neurons, flatten_neurons) .* 0.1))
wo = Variable{Matrix{Float64}}(output=(randn(output_neurons, hidden_neurons) .* 0.1))

function dense(w, x, activation)
    return activation(w * x)
end

function cross_entropy_loss(y, ŷ)
    return sum(-y .* log.(ŷ))
end

function relu(x)
    return max.(x, Constant(0))
end

function conv(w, x, activation)
    out = conv(x, w)
    return activation(out)
end

function network(x, wc, wh, wo, y)
    c = conv(wc, x, relu)
    m = maxpool(c, Constant(2))
    f = flatten(m)
    d1 = dense(wh, f, relu)
    d2 = dense(wo, d1, softmax)
    loss = cross_entropy_loss(y, d2)

    return topological_sort(loss), d2
end



function train(graph, input_x, input_y, y_output, epochs, optimizer, Wc, Wh, Wo)
    function runEpoch(e)
        train_loss = 0
        test_loss = 0
        train_accuracy = 0
        test_accuracy = 0

        println("Epoch: ", e)
        for j in 1:train_size
            @views input_x.output = train_data_x[:, :, :, j]
            @views input_y.output = train_data_y[j, :]

            train_loss += forward!(graph)
            train_accuracy += argmax(vec(y_output.output)) - 1 == train_y_outputs[j]
            backward!(graph)

            step!(optimizer, Wc, Wh, Wo)
        end
        println("Training complete")

        train_accuracy = train_accuracy / train_size

        println("Train accuracy: ", train_accuracy)

        for j in 1:test_size
            @views input_x.output = test_data_x[:, :, :, j]
            @views input_y.output = test_data_y[j, :]
            test_loss += forward!(graph)
            test_accuracy += argmax(vec(y_output.output)) - 1 == test_y_outputs[j]
        end
        test_accuracy = test_accuracy / test_size
        println("Test accuracy: ", test_accuracy)
    end

    for e = 1:epochs
        @time runEpoch(e)
    end
end

graph, y_output = network(input_x, wc, wh, wo, input_y)
@time train(graph, input_x, input_y, y_output, settings.epochs, optimize, wc, wh, wo)