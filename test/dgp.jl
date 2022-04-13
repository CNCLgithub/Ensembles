using Gen
using Random
using Ensembles

#function recurse(f::Function, x::T, n::Int64)::T where {T}
#   n === 0 ? x : recurse(f, f(x), n-1)
#end

function rep_test()
    Random.seed!(123)
    gm = RepulsionGM(n_dots = 6,
                    area_width = 600.0,
                    area_height = 600.0,
                    img_width = 512,
                    img_height = 512,
                    dot_repulsion= 1.)
    rdgp = RepulsionDGP(trials = 2,
                        k=240,
                        max_distance=Inf,
                        min_distance=-Inf,
                        tries=1e7,
                        out_dir = "/spaths/datasets/test")
    dgp(rdgp, gm)
end


println("performing simple test")
rep_test()
