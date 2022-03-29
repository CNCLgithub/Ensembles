using Gen
using Ensembles

#function recurse(f::Function, x::T, n::Int64)::T where {T}
#   n === 0 ? x : recurse(f, f(x), n-1)
#end

function rep_test()
    # TODO
    # first iniitalize a scene
    # then simulation for 10 steps
    gm = RepulsionGM(n_dots = 20,
                    area_width = 300.0,
                     area_height = 300.0,
                     wall_repulsion = 100.)
    rdgp = RepulsionDGP(trials = 10, k=20, out_dir = "/spaths/datasets/pilot")
    dgp(rdgp, gm; tries= 100)
end


println("performing simple test")
rep_test()
