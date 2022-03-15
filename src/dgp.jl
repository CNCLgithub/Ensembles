abstract type DGP end

@with_kw struct RepulsionDGP <: DGP
    # number of trials
    trials::Int64 = 5
    # number of steps per trial
    k::Int64 = 10
    # maximum distance between trackers for a valid step 
    max_distance::Float64 = 100.0
    # minimum distance between tracker for a valid step
    min_distance::Float64 = 20.0

    out_dir::String
end

function distances(objects::Vector{Dot})
    n = length(objects)
    # initialize distance matrix
    dmat = zeros(n, n)
    @inbounds for i = 1:n, j = 1:n
        # 0 distance if same object
        i === j  && continue
        # otherwise l2 distance
        i_pos = objects[i].pos
        j_pos = objects[j].pos
        dmat[i, j] =  norm(i_pos - j_pos)
    end
    dmat
end


function write_states(gm::RepulsionGM, states::Vector{RepulsionState}, path::String)
    t = length(states)
    positions = Array{Float64}(undef, t, 2, gm.n_dots)
    for i = 1:t, j = 1:gm.n_dots
        positions[i, :, j] = states[i].objects[j].pos[:]
    end
    data = Dict(positions = positions)
    json_path = "$(path)/states.json"
    open(json_path, "w") do f
        write(f, json(data))
    end
    return nothing
end

function write_graphics(gm::RepulsionGM, states::Vector{RepulsionState}, path::String)
    t = length(states)
    img_path = "$(path)/images"
    isdir(img_path) || mkpath(img_path)

    for i = 1:t
        #gstate = zeros(gm.img_dims)
        for i = 1:t, j = 1:gm.n_dots
          # see `write_states` above and the file `test/repulsion.jl`
            img_file = "$(img_path)/$(i).png"
            gstate = states[i].objects[j].gstate   
            save(img_file, gstate)
        end
        t[i] = sum/n_dots

        # see https://juliaimages.org/latest/function_reference/#ref_io
        # for how to save image
        save(img_file, gstate)
    end
    return nothing
end

# constraints for trial generation

"""

Returns `true` if all of the following are true:

- none of the objects overlap
"""
function initial_state_constraint(p::RepulsionDGP, gm::RepulsionGM, st::RepulsionState)::Bool
    # see implementation above
    #ds = distances(st.objects).- (gm.dot_radius * 2)

    # TODO make sure to incorporate the radius of each object
    # we can assume that the radius is fixed
    # objects dont overlap
    sum(ds .<= (gm.dot_radius * 2)) === size(ds, 1)
end

"""

Returns 'trues' if:

- objects are within a minimum proximity
"""
function step_constraint(p::RepulsionDGP, gm::RepulsionGM, st::RepulsionState)
    # used `p.max_distance`
    ds = distances(st.objects).- (gm.dot_radius * 2)
    #thresh = p.max_distance 
    #get the first element that's not 0 
    (sum(xs .< ((gm.dot_radius * 2) + min_distance)) === size(xs, 1)) & !(any( xs .> (max_distance + (gm.dot_radius * 2)))
end


function dgp(p::RepulsionDGP, gm::RepulsionGM; tries::Int64 = 100)
    if (tries <= 0):
        return nothing
    init_state = rpl_init(gm)

    if !initial_state_constraint(init_state)
        # overlapping, recursively try again
        # apply step and check distances
        # once passes, set all velocities to zero
        return dgp(gm, k; tries = tries - 1)
    end

    # initial state passes, begin generating steps
    states = Vector{RepulsionState}(undef, p.k)
    states[1] = init_state
    @inbounds for t = 2:p.k
        states[t] = Ensembles.step(gm, states[t - 1])
        # check to see if objects diverge too much
        if !step_constraint(gm, states[t])
            # restart if failed
            return dgp(gm, k; tries = tries - 1)
        end
    end

    # everything is passed
    # save state and images
    write_state(gm, states, path)
    write_graphics(gm, states, path)
end
