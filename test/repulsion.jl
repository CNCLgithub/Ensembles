using Gen
using Ensembles

# function recurse(f::Function, x::T, n::Int64)::T where {T}
#     n === 0 ? x : recurse(f, f(x), n-1)
# end

function rep_test()
    # TODO
    # first iniitalize a scene
    # then simulation for 10 steps
    gm = RepulsionGM()
    # https://www.gen.dev/dev/ref/gfi/#Gen.simulate
    init_state_tr = Gen.simulate(rpl_init, (gm,))
    # extract what we want from Gen.Trace (hint look at `rpl_init`)
    init_state = Gen.?(init_state_tr)
    current_state = init_state
    display(current_state) # pretty printing
    @show current_state # (expr -> val) : line number
    # evolve the state n steps
    # For fun, see if you can find a function like `recurse`
    for i = 1:10
        println("on step $(i)")
        current_state = Ensembles.step(gm, current_state)
        # gstate = zeros(gm.img_dims)
        # for j = 1:gm.n_dots
        #     dot = ?
        #     println("object ${j}")
        #     gstate .+= dot.gstate
        # end
        # gstate ./= gm.n_dots
        # display(gstate)
    end
end

println("performing simple test")
rep_test()
