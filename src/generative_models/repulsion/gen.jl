################################################################################
# Initial State
################################################################################
@gen static function rpl_tracker(gm::RepulsionGM)::Dot
    xs, ys = tracker_bounds(gm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(von_mises(0.0, 1e-5), :ang) # super flat
    mag = @trace(normal(vel, 1e-2), :std)

    vx = mag * cos(ang)
    vy = mag * sin(ang)

    pos = SVector{2, Float64}([x, y])
    vel = SVector{2, Float64}([vx, vy])
    return Dot(gm, pos, vel)
end


"""
Samples a random scene
"""
@gen static function rpl_init(gm::RepulsionGM)::RepulsionState
    gms = fill(gm, gm.n_dots)
    trackers = @trace(Gen.Map(rpl_tracker)(gms), :trackers)
    state = RepulsionState(gm, SVector{Dot}(trackers))
    return state
end
################################################################################
# Dynamics
################################################################################

@gen function rpl_step(st::RepulsionState, v::Int64)::Dot

    @unpack inertia, spring, sigma = gm

    _x, _y, _z = dot.pos
    vx, vy = dot.vel

    vx = @trace(normal(inertia * vx - spring * _x, sigma), :vx)
    vy = @trace(normal(inertia * vy - spring * _y, sigma), :vy)

    x = _x + vx
    y = _y + vy

    return Dot(pos=[x,y,_z], vel=[vx,vy] ww)
end


@gen function isr_update(prev_cg::CausalGraph)
    cg = deepcopy(prev_cg)
    vs = get_object_verts(cg, Dot)

    # first start with repulsion step (deterministic)
    things = isr_repulsion_step(cg)
    cg = dynamics_update(get_dm(cg), cg, things)

    # then brownian step (random)
    cgs = fill(cg, length(vs))
    things = @trace(Map(isr_step)(cgs, vs), :trackers)
    cg = dynamics_update(get_dm(cg), cg, things)

    return cg
end


@gen function isr_pos_kernel(t::Int,
                         prev_cg::CausalGraph)
    # advancing causal graph according to dynamics
    # (there is a deepcopy here)
    cg = @trace(isr_update(prev_cg), :dynamics)
    return cg
end


@gen function gm_isr_pos(k::Int, gm, dm)
    cg = get_init_cg(gm, dm)
    init_state = @trace(isr_init(cg), :init_state)
    states = @trace(Gen.Unfold(isr_pos_kernel)(k, init_state), :kernel)
    result = (init_state, states)
    return result
end
