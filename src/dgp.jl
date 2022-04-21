using UnicodePlots
using Images


export DGP, RepulsionDGP, dgp
abstract type DGP end

@with_kw struct RepulsionDGP <: DGP
    # number of trials
    trials::Int64 = 5
    # number of steps per trial
    k::Int64 = 10
    # maximum distance between trackers for a valid step
    max_distance::Float64 = Inf
    # minimum distance between tracker for a valid step
    min_distance::Float64 = 20.0

    tries::Int64 = 10000
    out_dir::String
end

function distances(objects::Vector{Dot})
    n = length(objects)
    # initialize distance matrix
    dmat = zeros(n, n)
    @inbounds for i = 1:n, j = 1:n
        # 0 distance if PNG_COLOR_TYPE_GRAYsame object
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
    positions = []
    state_path = "$(path)/serialized"
    isdir(state_path) || mkpath(state_path)
    for i = 1:t
        t_step = []
        for j = 1:gm.n_dots
            push!(t_step, states[i].objects[j].pos[:])
        end
        push!(positions, t_step)
    end
    data = Dict(:positions => positions)
    json_path = "$(path)/states.json"
    open(json_path, "w") do f
        write(f, JSON.json(data))
    end

    for i = 1:t
        gstate = zeros(gm.img_dims)
        for j = 1:gm.n_dots
            # see `write_states` above and the file `test/repulsion.jl`
            obj = states[i].objects[j]
            gstate = states[i].gstate[j]
            #replacing gstate of obj to be the old gstate
            obj = update(obj, obj.pos, obj.vel, gstate)
            json_state_path = "$(state_path)/$(i)_$(j).json"
            open(json_state_path, "w") do f
                write(f, JSON.json(obj))
            end
            # img_file = "$(img_path)/$(i)_$(j).png"
            # # see https://juliaimages.org/latest/function_reference/#ref_io
            # # for how to save image
            # # println(img_file)
            # # display(obj.gstate)
            # gstate .+= obj.gstate
            # save(img_file, obj.gstate)
        end
        # #gstate ./= gm.n_dots
        # clamp!(gstate,0.,1.)
        # scene_file = "$(img_path)/$(i).png"
        # save(scene_file, gstate)

    end

    return nothing
end

function write_graphics(gm::RepulsionGM, states::Vector{RepulsionState}, path::String)
    t = length(states)
    img_path = "$(path)/images"
    isdir(img_path) || mkpath(img_path)
    for i = 1:t
        gstate = zeros(gm.img_dims)
        for j = 1:gm.n_dots
            # see `write_states` above and the file `test/repulsion.jl`
            obj = states[i].objects[j]
            img_file = "$(img_path)/$(i)_$(j).png"
            # see https://juliaimages.org/latest/function_reference/#ref_io
            # for how to save image
            # println(img_file)
            # display(obj.gstate)
            gstate .+= obj.gstate
            save(img_file, obj.gstate)
        end
        #gstate ./= gm.n_dots
        clamp!(gstate,0.,1.)
        scene_file = "$(img_path)/$(i).png"
        save(scene_file, gstate)



    end
    return nothing
end

# constraints for trial generation

"""

Returns `true` if all of the following are true:

- none of the objects overlapreturn true
"""
function initial_state_constraint(p::RepulsionDGP, gm::RepulsionGM, st::RepulsionState)::Bool
    # see implementation above
    ds = distances(st.objects).- (gm.dot_radius * 2)

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
    (sum(ds .< ((gm.dot_radius * 2) + p.min_distance)) === size(ds, 1)) &
        !(any(ds .> (p.max_distance + (gm.dot_radius * 2))))
    return true

end


"""

Generates a single trial
"""
function dgp_trial(p::RepulsionDGP, gm::RepulsionGM, idx::Int64)

    init_state = rpl_init(gm)



    if !initial_state_constraint(p, gm, init_state)
        # overlapping, recursively try again
        # apply step and check distances
        # once passes, set all velocities to zero
        #return dgp_trial(p, gm, idx, tries - 1)
        return false
    end

    # initial state passes, begin generating steps
    states = Vector{RepulsionState}(undef, p.k)
    states[1] = init_state
    @inbounds for t = 2:p.k
        states[t] = Ensembles.step(gm, states[t - 1])
        # check to see if objects diverge too much
        if !step_constraint(p, gm, states[t])
            # restart if failed
            #return dgp_trial(p, gm, idx, tries - 1)
            return false
        end
    end
    # everything is passed
    # save state and images
    trial_path = "$(p.out_dir)/$(idx)"
    isdir(trial_path) || mkpath(trial_path)
    write_states(gm, states, trial_path)
    write_graphics(gm, states, trial_path)
    return true
end

function dgp(p::RepulsionDGP, gm::RepulsionGM)
    isdir(p.out_dir) || mkpath(p.out_dir)
    for i = 1:p.trials
        for _ = 1:p.tries
            dgp_trial(p, gm, i) && break
        end
    end
end
