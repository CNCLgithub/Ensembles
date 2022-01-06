export Space

"""
Projection of a Thing into an observation space
"""
const Space{T,N} = AbstractArray{T,N}

function render(cg::CausalGraph)::Diff
    gr = get_graphics(cg)
    spaces = render(gr, cg)
end

include("flow.jl")
include("masks.jl")
include("graphics_module.jl")
