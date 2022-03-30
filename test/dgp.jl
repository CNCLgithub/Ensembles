using Gen
using Random
using Ensembles

#function recurse(f::Function, x::T, n::Int64)::T where {T}
#   n === 0 ? x : recurse(f, f(x), n-1)
#end

function rep_test()
    Random.seed!(123)
    gm = RepulsionGM(n_dots = 2)
    rdgp = RepulsionDGP(trials = 1,
                        k=60,
                        out_dir = "/spaths/datasets/test")
    dgp(rdgp, gm)
end


println("performing simple test")
rep_test()
