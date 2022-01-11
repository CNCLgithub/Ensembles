struct Wall <: Thing
    # Dynamics
    d::Float64 # the distance from the center
    normal::SVector{2, Float64} # normal vector
    # Kinematics <none>
    # Graphcis <none>
end

struct Dot <: Thing
    # Dynamics
    radius::Float64 # usually around 20

    # Kinematics
    pos::SVector{2, Float64}
    vel::SVector{2, Float64}

    # Graphics
    gstate::SparseMatrixCSC{Float64} # graphics state
end


@with_kw struct RepulsionGM <: GenerativeModel

    # Epistemics
    n_dots::Int64 = 2


    # Dynamics
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 60.0
    rep_inertia::Float64 = 0.9

    #  Kinematics
    area_width::Float64 = 400.
    area_height::Float64 = 400.
    dimensions::SVector{2, Float64} = SVector([area_width, area_height])
    vel::Float64 = 10.0 # base velocity

    # Graphics
    img_width::Int64 = 100
    img_height::Int64 = 100
    img_dims::SVector{2, Int64} = SVector([img_width, img_height])
end

struct RepulsionState <: GMState
    walls::SVector{4, Wall}
    objects::AbstractArray{Dot, 1} # TODO: specify me, maybe SVector{Dot}`?
    # distances::SMatrix{Float64}
    # optional
end

function RepulsionState(gm::RepulsionGM, objects::SVector{Dot})
    walls = # TODO define me
    RepulsionState(walls, objects)
end


# function load(::Type{RepulsionGM}, path::String)
#     RepulsionGM(;read_json(path)...)
# end

function step(gm::RepulsionGM, state::RepulsionState)::RepulsionState

    # Dynamics (computing forces)
    # for each dot compute forces
    @unpack n_dots = gm
    @unpack walls, objects = state
    new_dots = Vector{Dot}(undef, n_dots)
    @inbounds for i = 1:n_dots
        facc = zeros(2) # force accumalator
        dot = objects[i]
        for w in state.walls
            force!(facc, w, dot)
        end
        # TODO add interaction with other dots

        # kinematics: resolve forces to pos vel
        (new_pos, new_vel) = update_kinematics(gm, dot, facc)
        # also do graphical update
        new_gstate = update_graphics(gm, dot, new_pos)
        new_dots[i] = update(dot, new_pos, new_vel, new_gstate)
    end

    RepulsionState(walls, SVector(new_dots))
end

"""Computes the force of A -> B"""
function force!(f::Vector{Float64}, ::Thing, ::Thing)
    error("Not implemented")
end
function force!(f::Vector{Float64}, w::Wall, d::Dot)
    # TODO
end
function force!(f::Vector{Float64}, a::Dot, b::Dot)
    # TODO
end

function update_kinematics(gm::Repulsion, d::Dot, f::Vector{Float64})
    # TODO
    (new_pos, new_vel)
end

function update_graphics(gm::Repulsion, d::Dot, new_pos::SVector{2, Float64})

    @unpack area_width, area_height = gm
    @unpack img_width, img_height = gm

    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos...,
                                 img_height, img_width,
                                 area_width, area_height)
    scaled_r = d.radius/area_width*img_width # assuming square

    # first "copy" over the previous state as a dense array
    dstate = Array(d.gstate)
    # decay img
    rmul!(dstate, gm.decay_rate)
    # write new pixels
    exp_dot_mask!(dstate, x, y, scaled_r, gr)
    gstate = sparse(dstate)
    droptol!(gstate, gr.min_mag)
    return gstate
end


function update(d::Dot, pos, vel, gstate)
    Dot(d.radius, pos, vel, gstate)
end

include("gen.jl")
include("helpers.jl")
