export RepulsionGM, step

using LinearAlgebra

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
    mass::Float64

    # Kinematics
    pos::SVector{2, Float64}
    vel::SVector{2, Float64}

    # Graphics
    gstate::SparseMatrixCSC{Float64} # graphics state
end

JSON.lower(d::Dot) = Dict(
    radius  => d.radius,
    mass => d.mass,
    pos => d.pos,
    vel => d.vel,
    gstate => d.gstate
)

@with_kw struct RepulsionGM <: GenerativeModel

    # EPISTEMICS
    n_dots::Int64 = 2


    # DYNAMICS
    dot_repulsion::Float64 = 0.01
    wall_repulsion::Float64 = 1.5
    distance::Float64 = 20.0
    dot_mass::Float64 = 1.0
    dot_radius::Float64 = 10.0

    # KINEMATICS
    area_width::Float64 = 300.0
    area_height::Float64 = 300.0
    dimensions::Tuple{Float64, Float64} = (area_width, area_height)
    #play around and increase vel
    vel::Float64 = 5.0 # base velocity


    # GRAPHICS
    img_width::Int64 = 128
    img_height::Int64 = 128
    img_dims::Tuple{Int64, Int64} = (img_width,img_height)
    decay_rate::Float64 = 0.0
    min_mag::Float64 = 1E-4
    inner_f::Float64 = 1.0
    outer_f::Float64 = 1.0
    inner_p::Float64 = 0.95
    outer_p::Float64 = 0.3
end


struct RepulsionState <: GMState
    walls::SVector{4, Wall}
    objects::Vector{Dot}
    gstate::Vector{SparseMatrixCSC{Float64}}
end

##Updated upstream
function RepulsionState(gm::RepulsionGM, dots)
    walls = init_walls(gm)
    empty_state = spzeros(dots[1].gstate)
    RepulsionState(walls, dots, fill(empty_state, length(dots)))
end

const WALL_ANGLES = [0, pi/2, pi, 3/2 * pi]

function init_walls(gm::RepulsionGM)
   ws = Vector{Wall}(undef, 4)
   d = [gm.dimensions[1] * .5, gm.dimensions[2] * .5, -gm.dimensions[1] * .5, -gm.dimensions[2]*.5]
   @inbounds for (i, theta) in enumerate(WALL_ANGLES)
   ## d should be constant;
        #v = [x,y]
        normal = [cos(theta), sin(theta)]
        ws[i] = Wall(d[i], normal)
    end
    return SVector{4, Wall}(ws)

end

function step(gm::RepulsionGM, state::RepulsionState)::RepulsionState

    # Dynamics (computing forces)
    # for each dot compute forces
    @unpack n_dots = gm
    @unpack walls, objects = state
    new_dots = Vector{Dot}(undef, n_dots)
    gstates = Vector{SparseMatrixCSC{Float64}}(undef, n)

    @inbounds for i = 1:n_dots
        facc = zeros(2) # force accumalator
        dot = objects[i]
        for w in state.walls
            force!(facc, gm, w, dot)
        end
        for j = 1:n_dots
            i==j && continue
            force!(facc, gm, dot, objects[j])
        end

        # TODO add interaction with other dots
        #
        # kinematics: resolve forces to pos vel
        (new_pos, new_vel) = update_kinematics(gm, dot, facc)
        # also do graphical update
        new_gstate = update_graphics(gm, dot, new_pos)
        new_dots[i] = update(dot, new_pos, new_vel, new_gstate)
        gstates[i] = dot.gstate
    end

    RepulsionState(walls, new_dots, gstates)
end

normalvec(w::Wall, pos) = w.normal

"""Computes the force of A -> B"""
function force!(f::Vector{Float64}, dm::RepulsionGM, ::Thing, ::Thing)
    error("Not implemented")
end
function force!(f::Vector{Float64}, dm::RepulsionGM, w::Wall, d::Dot)
    #multiply unit vector of wall by position of dot
    # d - norm|unit_vector * x|
    @unpack pos = d
    #normal vec is unit vector; using partial derivatives (derivative of l2 norm)
    #constant mass system w object interaction
    #unit_vector = w.normal/sqrt(w.normal[1]^2 + w.normal[2]^2)
    # n = abs(w.d + LinearAlgebra.norm(w.normal .* pos))
    n = LinearAlgebra.norm(w.normal .* pos + w.normal .* w.d)
    # absolute_force = dm.wall_repulsion*exp(n/(dm.distance^2))
    # absolute_force = n > dm.distance ? 0 : exp(log(1.0) - log(n) - log(dm.wall_repulsion))
    thresh = 2 * dm.distance
    absolute_force = (n > thresh) ? 0. : exp(dm.distance / (n * dm.wall_repulsion))
    f .+= absolute_force * (w.normal)
    return nothing

end

function force!(f::Vector{Float64}, dm::RepulsionGM, a::Dot, b::Dot)
    v = a.pos- b.pos
    norm_v = norm(v)
    thresh = 4*b.radius
    absolute_force = (norm_v > thresh) ? 0. : exp(dm.distance / (norm_v * dm.dot_repulsion))
    if norm_v>thresh
        absolute_force *= -1.0
    end
    f .+= absolute_force .* normalize(v)
    return nothing
end


function update_kinematics(gm::RepulsionGM, d::Dot, f::Vector{Float64})
    # treating force directly as velocity; update velocity by x percentage; but f isn't normalized to be similar to v
    a = f/d.mass
    new_vel = d.vel + a
    new_pos = clamp.(d.pos + new_vel, -gm.area_height, gm.area_height)
    return new_pos, new_vel
end


function update_graphics(gm::RepulsionGM, d::Dot, new_pos::SVector{2, Float64})

    @unpack area_width, area_height = gm
    @unpack img_width, img_height = gm

    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos...,
                                 img_height, img_width,
                                 area_width, area_height)
    scaled_r = d.radius/area_width*img_width # assuming square
    gstate = exp_dot_mask(x, y, scaled_r, img_width, img_height, gm)

    # Mario: trying to deal with segf when dropping
    decayed = deepcopy(d.gstate)
    rmul!(decayed, gm.decay_rate)
    droptol!(decayed, gm.min_mag)

    # overlay new render onto memory

    #without max, tail gets lost; . means broadcast element-wise
    max.(gstate, decayed)
    #map!(max, gstate, gstate, decayed)
    return gstate
end


function update(d::Dot, pos, vel, gstate)
    Dot(d.radius, d.mass, pos, vel, gstate)
end

function render(gm::RepulsionGM, st::RepulsionState)

    gstate = zeros(gm.img_dims)
    @inbounds for j = 1:gm.n_dots
        #dot = ?
        dot = current_state.objects[j]
        gstate .+= dot.gstate
    end
    rmul!(gstate, 1.0 / gm.n_dots)
    Gray.(gstate)
end

include("gen.jl")
include("helpers.jl")
