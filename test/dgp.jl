using Gen
using Ensembles
using UnicodePlots
export DGP, RepulsionDGP, dgp

#function recurse(f::Function, x::T, n::Int64)::T where {T}
#   n === 0 ? x : recurse(f, f(x), n-1)
#end

function rep_test()
    # TODO
    # first iniitalize a scene
    # then simulation for 10 steps
    gm = RepulsionGM(area_width = 300.0,
                     area_height = 300.0,
                     wall_repulsion = 100.)
    rdgp = RepulsionDGP(out_dir = "/spaths/datasets/pilot")

end

println("performing simple test")
rep_test()
