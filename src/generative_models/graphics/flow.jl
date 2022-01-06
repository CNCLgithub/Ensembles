abstract type Flow{T<:Space} end

evolve(::Flow, ::Space) = error("not implemented")

# with respect to one object
@with_kw struct ExponentialFlow{T} <: Flow{T}
    decay_rate::Float64
    exp_dr::Float64 = exp(decay_rate)
    memory::T
    lower::Float64 = 1E-7
end

function ExponentialFlow(flow::ExponentialFlow{T}, space::T) where {T <: Space}
    # decay memory
    decayed = flow.memory * flow.exp_dr
    droptol!(decayed, flow.lower)
    # overlay new render onto memory
    memory = max.(space, decayed)
    ExponentialFlow{T}(flow.decay_rate, flow.exp_dr,
                       memory, flow.lower)
end

evolve(flow::ExponentialFlow{T}, space::T) where {T <: Space} = ExponentialFlow(flow, space)
