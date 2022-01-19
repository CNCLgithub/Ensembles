
abstract type Thing end

struct Wall <: Thing
    # Dynamics
    d::Float64 # the distance from the center
    normal::SVector{2, Float64} # normal vector
    # Kinematics <none>
    # Graphcis <none>
end

struct Dot <: Thing
    # Dynamics
    radius::Float64

    # Kinematics
    pos::SVector{2, Float64}
    vel::SVector{2, Float64}

    # Graphics
    gstate::SMatrix{Float64} # graphics state
end


@with_kw struct RepulsionGM <: GenerativeModel

    # Epistemics
    n_dots::Int64 = 2


    # Dynamics
    dot_repulsion::Float64 = 80.0
    wall_repulsion::Float64 = 50.0
    distance::Float64 = 60.0
    rep_inertia::Float64 = 0.9

    # Kinematics
    dimensions::SVector{2, Float64} = SVector([100., 100.])
    vel::Float64 = 10.0 # base velocity

    # Graphics
    img_dims::SVector{2, Int64} = SVector([100, 100])
end

struct RepulsionState <: GMState
    walls::SVector{4, Wall}
    objects::SVector{Dot}
    distances::SMatrix{Float64}
end

function RepulsionState(gm::RepulsionGM, objects::SVector{Dot})
    walls = SVector{4,Wall}# define me
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

normalvec(w::Wall, pos) = w.normal

"""Computes the force of A -> B"""
function force!(f::Vector{Float64}, ::Object, ::Object)
    error("Not implemented")
end
function force!(f::Vector{Float64}, w::Wall, d::Dot, dm::RepulsionGM)
    @unpack pos = d
    n = normalvec(w, p.pos)
    #pos_proj = w -> _project(pos, w.p1, w.p2))

    v = dot(w.d - p.pos, n)
    nv = norm(v)
    absolute_force = dm.wall_repulsion*exp(nv/(dm.distance^2))
    f += (absolute_force / nv) .* v

    return force

end

function force!(f::Vector{Float64}, a::Dot, b::Dot, dm::RepulsionGM)
    @unpack pos = a
    @unpack other_pos = b
    force = zeros(2)
    for j = 1:length(other_pos)
        v = pos - other_pos[j]
        nv = norm(v)
        absolute_force = dm.dot_repulsion*exp(nv/(dm.distance^2))
        f += (absolute_force / nv) .* v
    end
    return force
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