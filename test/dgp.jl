using Gen
using Random
using Ensembles
using JSON
#function recurse(f::Function, x::T, n::Int64)::T where {T}
#   n === 0 ? x : recurse(f, f(x), n-1)
#end

function rep_test()
    Random.seed!(123)
    gm = RepulsionGM(n_dots = 6,
                    area_width = 600.0,
                    area_height = 600.0,
                    img_width = 128,
                    img_height = 128,
                    dot_repulsion= 1.)
    rdgp = RepulsionDGP(trials = 2,
                        k=240,
                        max_distance=Inf,
                        min_distance=-Inf,
                        tries=1e7,
                        out_dir = "/spaths/datasets/test")
    m = Dict(:n_dots=>gm.n_dots,
    :area_width=>gm.area_width,
    :area_height=>gm.area_height,
    :img_width=>gm.img_width,
    :img_height=>gm.img_height,
    :trials=>gm.trials,
    :k=>gm.k)

    mpath = "/spaths/datasets/test_manifest.json"
    open(mpath, "w") do f
        write(f, JSON.json(m))
    end

    dgp(rdgp, gm)
end


println("performing simple test")
rep_test()
