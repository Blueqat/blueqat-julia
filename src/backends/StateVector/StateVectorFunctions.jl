module StateVectorFunctions
    include("./StateVectorGates.jl")
    using Reexport
    @reexport using .StateVectorGates
    export apply!

    """
    X gate
    """
    function apply!(gate::X, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i )
            t = qubits[i0]
            @inbounds qubits[i0] = qubits[i0 + (1 << (target))]
            @inbounds qubits[i0 + (1 << (target))] = t
        end
    end

    """
    Y gate
    """
    function apply!(gate::Y, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0] * 1im
            @inbounds qubits[i0] = qubits[i0 + (1 << target)] * -1im
            @inbounds qubits[i0 + (1 << target)] = t
        end
    end

    """
    Z gate
    """
    function apply!(gate::Z, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            @inbounds qubits[_shifted(lower_mask, i) + (1 << target)] *= -1
        end
    end

    """
    H gate
    """
    function apply!(gate::H, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        sqrt2_inv = sqrt(0.5)
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0]
            u = qubits[i0 + (1 << target)]
            @inbounds qubits[i0] = (t + u) * sqrt2_inv
            @inbounds qubits[i0 + (1 << target)] = (t - u) * sqrt2_inv
        end
    end

    function _diaggate(qubits::Array{ComplexF64, 1}, n_qubits::Int64, target::Int64, factor::ComplexF64)
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i1 = _shifted(lower_mask, i) + (1 << target)
            @inbounds qubits[i1] *= factor
        end
    end

    """
    S gate
    """
    function apply!(gate::S, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = gate._target
        target = n_qubits - target
        factor = 1im
        _diaggate(qubits, n_qubits, target, factor)
    end

    """
    T gate
    """
    function apply!(gate::T, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = gate._target
        target = n_qubits - target
        factor = _exp(0.25im * pi)
        _diaggate(qubits, n_qubits, target, factor)
    end

    """
    RX gate
    """
    function apply!(gate::RX, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = gate._target
        target = n_qubits - target
        ang = gate._theta * 0.5
        angpi = ang / pi
        _cos = cospi(angpi)
        nisin = sinpi(angpi) * -1im
        lower_mask = (1 << target) - 1
        @simd for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0]
            u = qubits[i0 + (1 << target)]
            @inbounds qubits[i0] = _cos * t + nisin * u
            @inbounds qubits[i0 + (1 << target)] = nisin * t + _cos * u
        end
    end

    """
    RZ gate
    """
    function apply!(gate::RZ, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        ang = gate._theta * 0.5
        eit = _exp(1im * ang)
        eitstar = conj(eit)
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0]
            u = qubits[i0 + (1 << target)]
            @inbounds qubits[i0] *= eitstar
            @inbounds qubits[i0 + (1 << target)] *= eit
        end
    end

    """
    RY gate
    """
    function apply!(gate::RY, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        ang = gate._theta * 0.5
        angpi = ang / pi
        _cos = cospi(angpi)
        _sin = sinpi(angpi)
        lower_mask = (1 << target) - 1
        @simd for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0]
            u = qubits[i0 + (1 << target)]
            @inbounds qubits[i0] = _cos * t - _sin * u
            @inbounds qubits[i0 + (1 << target)] = _sin * t + _cos * u
        end
    end

    """
    U3 gate
    """
    function apply!(gate::U3, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = n_qubits - gate._target
        theta = gate._theta * 0.5
        thetapi = theta / pi
        phipi = gate._phi / pi
        lambdpi = gate._lambd / pi
        _cos = cos(thetapi)
        _sin = sin(thetapi)
        expadd = _exp((phipi + lambdpi) * 0.5im)
        expsub = _exp((phipi - lambdpi) * 0.5im)
        a = conj(expadd) * _cos
        b = conj(-expsub) * _sin
        c = expsub * _sin
        d = expadd * _cos
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            t = qubits[i0]
            u = qubits[i0 + (1 << target)]
            @inbounds qubits[i0] = a * t + b * u
            @inbounds qubits[i0 + (1 << target)] = c * t + d * u
        end
    end
    
    """
    CX gate
    """
    function apply!(gate::CX, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        c_mask = 0
        c_mask |= 1 << control
        t_mask = 1 << target
        n_loop = 1 << (n_qubits - 2)
        masks = _create_masks([control, target])
        for i in 1:n_loop
            i10 = _mult_shifted(masks, i-1) | c_mask
            i11 = i10 | t_mask
            i10 += 1
            i11 += 1
            t = qubits[i10]
            @inbounds qubits[i10] = qubits[i11]
            @inbounds qubits[i11] = t
        end
    end

    """
    CZ gate
    """
    function apply!(gate::CZ, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        all1 = 0
        for b in [control, target]
            all1 |= 1 << b
        end
        n_loop = 1 << (n_qubits - length([control, target]))
        masks = _create_masks([control, target])
        @simd for i in 1:n_loop
            i11 = _mult_shifted(masks, i-1) | all1
            @inbounds qubits[i11 + 1] *= -1
        end
    end

    """
    CRX gate
    """
    function apply!(gate::CRX, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        ang = gate._theta * 0.5
        angpi = ang / pi
        _cos = cospi(angpi)
        nisin = sinpi(angpi) * -1.0im
        c_mask = 0
        c_mask |= 1 << control
        t_mask = 1 << target
        n_loop = 1 << (n_qubits - length([control, target]))
        masks = _create_masks([control, target])
        for i in 1:n_loop
            i10 = _mult_shifted(masks, i-1) | c_mask
            i11 = i10 | t_mask
            i10 += 1
            i11 += 1
            t = qubits[i10]
            u = qubits[i11]
            @inbounds qubits[i10] = _cos * t + nisin * u
            @inbounds qubits[i11] = nisin * t + _cos * u
        end
    end

    """
    CRY gate
    """
    function apply!(gate::CRY, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        ang = gate._theta * 0.5
        angpi = ang / pi
        _cos = cospi(angpi)
        _sin = sinpi(angpi)
        c_mask = 0
        c_mask |= 1 << control
        t_mask = 1 << target
        n_loop = 1 << (n_qubits - length([control, target]))
        masks = _create_masks([control, target])
        for i in 1:n_loop
            i10 = _mult_shifted(masks, i-1) | c_mask
            i11 = i10 | t_mask
            i10 += 1
            i11 += 1
            t = qubits[i10]
            u = qubits[i11]
            @inbounds qubits[i10] = _cos * t - _sin * u
            @inbounds qubits[i11] = _sin * t + _cos * u
        end
    end

    """
    CRZ gate
    """
    function apply!(gate::CRZ, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        ang = gate._theta * 0.5
        eit = _exp(1.0im * ang)
        eitstar = conj(eit)
        c_mask = 0
        c_mask |= 1 << control
        t_mask = 1 << target
        n_loop = 1 << (n_qubits - length([control, target]))
        masks = _create_masks([control, target])
        for i in 1:n_loop
            i10 = _mult_shifted(masks, i-1) | c_mask
            i11 = i10 | t_mask
            i10 += 1
            i11 += 1
            @inbounds qubits[i10] *= eitstar
            @inbounds qubits[i11] *= eit
        end
    end

    """
    CP gate
    """
    function apply!(gate::CP, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        control = n_qubits - gate._control
        target = n_qubits - gate._target
        eit = _exp(1.0im * gate._theta)
        c_mask = 0
        c_mask |= 1 << control
        t_mask = 1 << target
        n_loop = 1 << (n_qubits - length([control, target]))
        masks = _create_masks([control, target])
        for i in 1:n_loop
            i11 = _mult_shifted(masks, i-1) | c_mask | t_mask + 1
            @inbounds qubits[i11] *= eit
        end
    end

    """
    Measurement gate
    """
    function apply!(gate::M, qubits::Array{ComplexF64, 1}, n_qubits::Int64)
        target = gate._target
        _rand = rand(Float64, 1)[1]
        p0 = _p0calc(qubits, target, n_qubits)
        if _rand < p0
            qubits = _reduce0(qubits, target, n_qubits, p0)
            gate._result = 0
        else
            qubits = _reduce1(qubits, target, n_qubits, p0)
            gate._result = 1
        end
    end

    function _p0calc(qubits::Array{ComplexF64, 1}, target::Int64, n_qubits::Int64)
        target = n_qubits - target
        p0 = 0.0
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            val = qubits[_shifted(lower_mask, i)]
            p0 += real(val) * real(val) + imag(val) * imag(val)
        end
        return p0
    end

    function _reduce0(qubits::Array{ComplexF64, 1}, target::Int64, n_qubits::Int64, p0::Float64)
        target = n_qubits - target
        sqrtp_inv = 1.0 / sqrt(p0)
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            qubits[i0] *= sqrtp_inv
            qubits[i0 + (1 << target)] = 0.0
        end
        return qubits
    end

    function _reduce1(qubits::Array{ComplexF64, 1}, target::Int64, n_qubits::Int64, p0::Float64)
        target = n_qubits - target
        sqrtp_inv = 1.0 / sqrt(1.0 - p0)
        lower_mask = (1 << target) - 1
        for i in 1:(1 << (n_qubits - 1))
            i0 = _shifted(lower_mask, i)
            qubits[i0 + (1 << target)] *= sqrtp_inv
            qubits[i0] = 0.0
        end
        return qubits
    end

    function _create_masks(indices::Array{Int64, 1})
        sort!(indices)
        masks = zeros(Int64, length(indices) + 1)
        for (i, x) in enumerate(indices)
            masks[i] = (1 << (x - (i - 1))) - 1
        end
        masks[length(masks)] = ~0
        for i in reverse(1:length(indices))
            masks[i + 1] &= ~masks[i]
        end
        return masks
    end

    function _shifted(lower_mask::Int64, idx::Int64)
        idx = idx - 1
        lower = idx & lower_mask
        higher = (idx & ~lower_mask) << 1
        return higher + lower + 1
    end
    
    function _mult_shifted(masks::Array{Int64, 1}, idx::Int64)
        shifted = 0
        for (i, x) in enumerate(masks)
            shifted |= (idx & x) << (i-1)
        end
        return shifted
    end

    function _exp(theta::Float64)
        thetapi = theta / pi / im
        res = cospi(thetapi) + im * sinpi(thetapi)
        return res
    end
    
    function _exp(theta::Complex{Float64})
        thetapi = real(theta / pi / im)
        res = cospi(thetapi) + im * sinpi(thetapi)
        return res
    end
end