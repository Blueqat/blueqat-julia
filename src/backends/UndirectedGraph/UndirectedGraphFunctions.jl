module UndirectedGraphFunctions
    include("./UndirectedGraphGates.jl")
    using Reexport
    @reexport using .UndirectedGraphGates
    export create_buckets, contract_graph, gate_parser

    mutable struct StateTensor
        item::Array{ComplexF64, 1}
        target::Int
        worldline1::Int
    end
    
    struct Tensor
        indices::Array{Int, 1}
        item::Array{ComplexF64}
    end

    function create_buckets(_n_qregs::Int, gates::Array{UG_Gate,1}, input_state::Array{Int,1}, output_state::Array{Int,1})
        worldlines = ones(Int64, _n_qregs)
        elements = []
        for i in 1:length(gates)
            obj, worldlines = get_tensor(gates[i], worldlines)
            push!(elements, obj)
        end

        num_variables = sum(worldlines)
        buckets = []
        for i in 1:sum(num_variables)
            push!(buckets, [])
        end

        for element in elements
            if typeof(element) == OneQubitGateTensor
                element.worldline1 += sum(worldlines[1:(element.target-1)])
                element.worldline2 += sum(worldlines[1:(element.target-1)])
            elseif typeof(element) == TwoQubitGateTensor
                element.worldline1 += sum(worldlines[1:(element.control-1)])
                element.worldline2 += sum(worldlines[1:(element.target-1)])
            end
        end

        for qreg in 1:_n_qregs
            input_val = convert(Array{ComplexF64, 1}, [1 - input_state[qreg], input_state[qreg]])
            _input_state = StateTensor(input_val, qreg, 1 + sum(worldlines[1:(qreg-1)]))
            insert!(elements, qreg, _input_state)
            output_val = convert(Array{ComplexF64, 1}, [1 - output_state[qreg], output_state[qreg]])
            _output_state = StateTensor(output_val, qreg, sum(worldlines[1:qreg]))
            push!(elements, _output_state)
        end

        for element in elements
            tensor = make_tensor(element)
            first_index = element.worldline1
            push!(buckets[first_index], tensor)
        end
        return buckets        
    end

    function get_tensor(gate::T, worldlines::Array{Int, 1}) where T <: UG_OneQubitGate
        mat = gate_mat(gate)
        worldline = worldlines[gate._target]
        obj = OneQubitGateTensor(mat, gate._target, worldline, worldline + 1)
        worldlines[gate._target] = worldline + 1
        return obj, worldlines
    end

    function get_tensor(gate::T, worldlines::Array{Int, 1}) where T <: UG_TwoQubitGate
        mat = gate_mat(gate)
        target1 = min(gate._control, gate._target)
        target2 = max(gate._control, gate._target)
        worldline1 = worldlines[target1]
        worldline2 = worldlines[target2]
        obj = TwoQubitGateTensor(mat, target1, target2, worldline1, worldline2)
        return obj, worldlines
    end

    function make_tensor(arr::T) where T <: UndirectedGraphTensor
        index1 = arr.worldline1
        index2 = arr.worldline2
        A = Tensor([index1, index2], arr.item)
        return A
    end

    function make_tensor(arr::StateTensor)
        index = arr.worldline1
        A = Tensor([index], arr.item)
        return A
    end

    function process_bucket(bucket)
        res = bucket[1]
        l = length(bucket)
        for tensor in bucket[2:l]
            res = broadcast_product(res, tensor)
        end
        return reduce_sum(res)
    end

    # element-wise product with broadcast, in the case size of each dim = 2
    function broadcast_product(A::Tensor, B::Tensor)
        indA = A.indices
        indB = B.indices
        indOut = sort(collect(union(Set(indA), Set(indB))))
        _indA = [Int(indOut[i] in Set(indA)) for i in 1:length(indOut)]
        _indB = [Int(indOut[i] in Set(indB)) for i in 1:length(indOut)]
        reshapeA = Tuple(_indA .+ 1)
        reshapeB = Tuple(_indB .+ 1)
        expandA = Tuple((_indA .- 1) .* -1 .+ 1)
        expandB = Tuple((_indB .- 1) .* -1 .+ 1)
        _A = repeat(reshape(A.item, reshapeA), outer=expandA)
        _B = repeat(reshape(B.item, reshapeB), outer=expandB)
        return Tensor(indOut, _A .* _B)
    end

    function reduce_sum(A::Tensor)
        arr = A.item
        l = length(A.indices)
        _arr = sum(A.item, dims=1)
        if length(size(_arr)) > 1
            _arr = dropdims(_arr; dims=1)
        end
        return Tensor(A.indices[2:l], _arr)
    end

    function contract_graph(buckets)
        result = 0
        for bucket in buckets
            if length(bucket) > 0
                tensor = process_bucket(bucket)
                if max(size(tensor.item)...) > 1
                    first_index = tensor.indices[1]
                    to_bucket = buckets[first_index]
                    push!(to_bucket, tensor)
                else
                    if result != 0
                        result = result .* tensor.item
                    else
                        result = tensor.item
                    end
                end
            end
        end
        return result[1]
    end

    function gate_mat(gate::X)
        return [0 1; 1 0]
    end

    function gate_mat(gate::Y)
        return [0 im; -im 0]
    end

    function gate_mat(gate::Z)
        return [1 0; 0 -1]
    end

    function gate_mat(gate::H)
        return 1 / sqrt(2) * [1 1; 1 -1]
    end

    function gate_mat(gate::S)
        return [1 0; 0 _exp(im*pi/2)]
    end

    function gate_mat(gate::T)
        return [1 0; 0 _exp(im*pi/4)]
    end

    function gate_mat(gate::CZ)
        return [1 1; 1 -1]
    end

    function gate_mat(gate::RX)
        theta = gate._theta
        thetapi = theta / pi
        return [cospi(thetapi/2) -im*sinpi(thetapi/2); -im*sinpi(thetapi/2) cospi(thetapi/2)]
    end

    function gate_mat(gate::RY)
        theta = gate._theta
        thetapi = theta / pi
        return [cospi(thetapi/2) sinpi(thetapi/2); -sinpi(thetapi/2) cospi(thetapi/2)]
    end

    function gate_mat(gate::RZ)
        theta = gate._theta
        return [_exp(-im*theta/2) 0; 0 _exp(im*theta/2)]
    end

    function gate_parser(gate::CX)
        parsed_gates = []
        push!(parsed_gates, H(gate._target))
        push!(parsed_gates, CZ(gate._control, gate._target))
        push!(parsed_gates, H(gate._target))
        return parsed_gates
    end

    function gate_parser(gate::T) where T <: UG_Gate
        return [gate]
    end

    function _exp(theta::Float64)
        thetapi = theta / pi / im
        res = cospi(thetapi) + im * sinpi(thetapi)
        return res
    end

end