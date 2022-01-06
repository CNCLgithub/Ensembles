using UnicodePlots
export Graphics

################################################################################
# Graphics
################################################################################

@with_kw struct Graphics <: AbstractGraphics
    img_width::Int64
    img_height::Int64
    img_dims::Tuple{Int64, Int64} = (img_width, img_height)
    flow_decay_rate::Float64
    inner_f::Float64
    inner_p::Float64
    outer_f::Float64
    outer_p::Float64
    nlog_bernoulli::Float64 = -100
    bern_existence_prob::Float64 = -expm1(nlog_bernoulli)
end

"""
    loads from JSON which has to have all the symboled elements
"""
function load(::Type{Graphics}, path::String)
    data = read_json(path)
    Graphics(; data...)
end

function predict(gr::Graphics, cg::CausalGraph)::Diff
    vs = collect(filter_vertices(cg, :space))
    nvs = length(vs)
    es = RFSElements{BitMatrix}(undef, nvs)
    @inbounds for j in 1:nvs
        es[j] = predict(gr, cg, vs[j], get_prop(cg, vs[j], :object))
    end
    Diff(Dict{ChangeDiff, Any}((:es => :es) => es))
end


################################################################################
# Rendering
################################################################################
"""
`Graphics` does not define any non-self graphical interactions.

Only updates local memory (:flow) and observation space (:space)
"""
function render(gr::Graphics,
                cg::CausalGraph)::Diff
    ch = ChangeDict()
    for v in LightGraphs.vertices(cg)
        @>> get_prop(cg, v, :object) render_elem!(ch, gr, cg, v)
    end
    Diff(Thing[], Int64[], StaticPath[], ch)
end

"""
Catch for undefined graphics
"""
function render_elem(::Graphics,
                     ::Thing)
    return nothing
end

"""
Rendering Dots
"""
function render_elem(gr::Graphics,
                     d::Dot)

    @unpack img_width, img_height = gr
    @unpack area_width, area_height = (get_prop(cg, :gm))

    # going from area dims to img dims
    x, y = translate_area_to_img(d.pos...,
                                 img_height, img_width,
                                 area_width, area_height)
    scaled_r = d.radius/area_width*img_width # assuming square

    gstate = d.gstate * gr.decay_rate
    droptol!(gstate, gr.min_mag)
    exp_dot_mask!(gstate, x, y, scaled_r, gr)
end

"""
Rendering `UniformEnsemble`

No `:flow` needed.
"""
function render_elem!(ch::ChangeDict,
                      gr::Graphics,
                      cg::CausalGraph,
                      v::Int64,
                      e::UniformEnsemble)
    @unpack img_dims = gr
    ch[v => :space] = Fill(e.pixel_prob, reverse(img_dims))
end

################################################################################
# Prediction
################################################################################
function predict(gr::Graphics, cg::CausalGraph, v::Int64, e::Dot)
    @unpack nlog_bernoulli = gr
    space = get_prop(cg, v, :space)
    LogBernoulliElement{BitMatrix}(nlog_bernoulli, mask, (space,))
end

function predict(gr::Graphics, cg::CausalGraph, v::Int64, e::UniformEnsemble)
    space = get_prop(cg, v, :space)
    PoissonElement{BitMatrix}(e.rate, mask, (space,))
end


################################################################################
# Helpers
################################################################################

function render_from_cgs(gr::Graphics,
                         gm::GMParams,
                         cgs::Vector{CausalGraph})
    k = length(cgs)
    # time x thing
    # first time step is initialization (not inferred)
    bit_masks= Vector{Vector{BitMatrix}}(undef, k-1)

    # initialize graphics
    g = first(cgs)
    set_prop!(g, :gm, gm)
    gr_diff = render(gr, g)
    @inbounds for t = 2:k
        g = cgs[t]
        set_prop!(g, :gm, gm)
        # carry over graphics from last step
        patch!(g, gr_diff)
        # render graphics from current step
        gr_diff = render(gr, g)
        patch!(g, gr_diff)
        @>> g predict(gr) patch!(g)
        # create masks
        bit_masks[t - 1] = rfs(get_prop(g, :es))
    end
    bit_masks
end

function triangular_dot_mask(x0::Float64, y0::Float64,
                             r::Float64, gr::Graphics)
    @unpack img_width, img_height = gr
    @unpack inner_p, inner_f, outer_p, outer_f = gr
    triangular_dot_mask(x0, y0, r, img_width, img_height,
                        outer_f, inner_f, outer_p, inner_p)
end

function exp_dot_mask(x0::Float64, y0::Float64,
                      r::Float64, gr::Graphics)
    @unpack img_width, img_height = gr
    @unpack inner_p, inner_f, outer_p, outer_f = gr
    exp_dot_mask(x0, y0, r, img_width, img_height,
                        outer_f, inner_f, outer_p, inner_p)
end
