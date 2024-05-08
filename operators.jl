import Base: *, -, sum, log, max
import LinearAlgebra: mul!
include("structures_graph_pass.jl")

*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = A * x

backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) =
    let
        one = ones(length(node.output))
        Jx = diagm(vec(y) .* one)
        Jy = diagm(x .* one)
        tuple(Jx' * g, Jy' * g)
    end

-(x::GraphNode) = ScalarOperator(-, x)
forward(::ScalarOperator{typeof(-)}, x) = -x
backward(::ScalarOperator{typeof(-)}, x, g) = tuple(-g .* one.(x))

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = tuple(g, g)

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) =
    let
        one = ones(length(vec(x)))
        J = one'
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) =
    let
        one = ones(length(node.output))
        Jx = diagm(one ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = log.(x)
backward(::BroadcastedOperator{typeof(log)}, x, g) = tuple(g ./ x)

Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) =
    let
        y = vec(node.output)
        J = diagm(y) .- y * y'
        tuple(J' * g)
    end

relu(x::GraphNode) = BroadcastedOperator(relu, x)
forward(::BroadcastedOperator{typeof(relu)}, x) = max.(x, 0)
backward(::BroadcastedOperator{typeof(relu)}, x, g) = tuple(g .* (x .> 0))

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = reshape(x, :, 1)
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = reshape(g, size(x))


conv(x::GraphNode, w::GraphNode) = BroadcastedOperator(conv, x, w)

function forward(::BroadcastedOperator{typeof(conv)}, x, w)
    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)

    cv_dt = (;
        p=0, #padding
        st=1, #stride
        out_h=Int(floor((H + 2 * 0 - FH) / 1)) + 1,
        out_w=Int(floor((W + 2 * 0 - FW) / 1)) + 1
    )

    x_pad = zeros(H + 2 * cv_dt.p, W + 2 * cv_dt.p, C)

    x_pad[cv_dt.p+1:end-cv_dt.p, cv_dt.p+1:end-cv_dt.p, :] = x
    out = zeros(cv_dt.out_h, cv_dt.out_w, K, 1)
    for i ∈ 1:cv_dt.out_h
        for j ∈ 1:cv_dt.out_w
            r_field =
                x_pad[(i-1)*cv_dt.st+1:(i-1)*cv_dt.st+FH, (j-1)*cv_dt.st+1:(j-1)*cv_dt.st+FW, :, :]

            r_field_flat = reshape(r_field, FH * FW * C, :)
            w_flat = reshape(w, FH * FW * C, K)
            out[i, j, :] = sum(w_flat .* r_field_flat, dims=1)
        end
    end
    return out
end
function backward(::BroadcastedOperator{typeof(conv)}, x, w, g)
    padding = 0
    stride = 1
    (H, W, C) = size(x)
    (FH, FW, _, K) = size(w)
    out_h = Int(floor((H + 2 * padding - FH) / stride)) + 1
    out_w = Int(floor((W + 2 * padding - FW) / stride)) + 1
    p = padding
    x_pad = zeros(H + 2p, W + 2p, C)
    x_pad[p+1:end-p, p+1:end-p, :] = x
    gx_pad = zeros(H + 2p, W + 2p, C)
    gw = zeros(size(w))

    for i ∈ 1:out_h
        for j ∈ 1:out_w
            r_field =
                x_pad[(i-1)*stride+1:(i-1)*stride+FH, (j-1)*stride+1:(j-1)*stride+FW, :, :]

            r_field_flat = reshape(r_field, FH * FW * C, :)
            w_flat = reshape(w, FH * FW * C, K)
            dout_local = reshape(g[i, j, :], K, 1)
            field_dout_prod = r_field_flat * dout_local'
            field_dout_prod = reshape(field_dout_prod, FH, FW, C, K)
            gw += field_dout_prod
            flat_dout_prod = w_flat * dout_local
            flat_dout_prod = reshape(flat_dout_prod, FH, FW, C, :)
            gx_pad[(i-1)*stride+1:(i-1)*stride+FH, (j-1)*stride+1:(j-1)*stride+FW, :, :] +=
                flat_dout_prod
        end
    end
    gx = gx_pad[p+1:end-p, p+1:end-p, :]
    return tuple(gx, gw)
end

maxpool(x::GraphNode, n::Constant) = BroadcastedOperator(maxpool, x, n)
function forward(node::BroadcastedOperator{typeof(maxpool)}, x, n)
    let
        M, N, C = size(x)
        M_out = 1 + (M - n) ÷ n
        N_out = 1 + (N - n) ÷ n
        out = zeros(Float64, M_out, N_out, C)
        for c = 1:C
            for i = 1:n:M
                for j = 1:n:N
                    @views x_view = x[i:(i+n-1), j:(j+n-1), c]
                    out[1+i÷n, 1+j÷n, c] = maximum(x_view)
                end
            end
        end
        out
    end
end

function backward(::BroadcastedOperator{typeof(maxpool)}, x, n, g)
    let
        M, N, C = size(x)
        M_out, N_out, _ = size(g)
        dx = zeros(Float64, M, N, C)
        for c = 1:C, i = 1:M_out, j = 1:N_out
            @views pool = x[1+(i-1)*n:i*n, 1+(j-1)*n:j*n, c]
            mask = (pool .== maximum(pool))
            dx[1+(i-1)*n:i*n, 1+(j-1)*n:j*n, c] = mask * g[i, j, c]
        end
        tuple(dx)
    end
end

