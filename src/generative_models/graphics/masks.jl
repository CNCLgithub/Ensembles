export get_masks,
        draw_dot_mask,
        draw_gaussian_dot_mask,
        translate_area_to_img


# Draws a dot mask, i.e. a BitMatrix
function draw_dot_mask(pos::Vector{T},
                       r::T,
                       w::I, h::I,
                       aw::T, ah::T) where {I<:Int64,T<:Float64}

    x, y = translate_area_to_img(pos[1], pos[2], w, h, aw, ah)
    mask = BitMatrix(zeros(h, w))
    radius = round(r * w / aw; digits = 3)
    draw_circle!(mask, [x,y], radius, true)
    return mask
end


# 2d gaussian function
function two_dimensional_gaussian(x::Int64, y::Int64, x_0::Float64, y_0::Float64, A::Float64,
                                  sigma_x::Float64, sigma_y::Float64)
    d = sigma_x * sigma_y
    nc = 1.0 / sqrt(two_pi_sqr * d)
    nc * exp(-( (x-x_0)^2/(2*sigma_x^2) + (y-y_0)^2/(2*sigma_y^2)))
end



"""
drawing a gaussian dot with two components:
1) just a dot at the center with probability 1 and 0 elsewhere
2) spread out gaussian modelling where the dot is likely to be in some sense
    and giving some gradient if the tracker is completely off
"""
function draw_gaussian_dot_mask(center::Vector{Float64},
                                r::Float64, w::Int64, h::Int64,
                                gauss_r_multiple::Float64,
                                gauss_amp::Float64,
                                gauss_std::Float64)
    scaled_sd = r * gauss_std
    threshold = r * gauss_r_multiple
    # mask = zeros(h, w) # mask is initially zero to take advantage of sparsity
    x0, y0 = center
    xlim = round.(Int64, [x0 - threshold, x0 + threshold])
    ylim = round.(Int64, [y0 - threshold, y0 + threshold])
    xbounds = clamp.(xlim, 1, w)
    ybounds = clamp.(ylim, 1, h)
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]
    for idx in CartesianIndices((xbounds[1]:xbounds[2],
                                 ybounds[1]:ybounds[2]))
        i,j = Tuple(idx)
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        (dst > threshold) && continue
        # v = two_dimensional_gaussian(i, j, x0, y0, gauss_amp, scaled_sd, scaled_sd)
        v = (dst <= scaled_sd ) ? gauss_amp : 0.01
        # flip i and j in mask
        push!(Is, j)
        push!(Js, i)
        push!(Vs, v)
    end
    sparse(Is, Js, Vs, h, w)
end

const ln_hlf = log(0.5)

#
# BenchmarkTools.Trial: 24 samples with 1 evaluation.
#  Range (min … max):   52.214 ms … 111.207 s  ┊ GC (min … max):  0.00% … 6.99%
#  Time  (median):     141.083 ms              ┊ GC (median):    12.06%
#  Time  (mean ± σ):      5.124 s ±  22.611 s  ┊ GC (mean ± σ):   7.00% ± 5.37%

#   █
#   █▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▁
#   52.2 ms         Histogram: frequency by time           111 s <

#  Memory estimate: 11.02 MiB, allocs estimate: 84393
function exp_dot_mask(x0::Float64, y0::Float64,
                      r::Float64,
                      w::Int64, h::Int64,
                      outer_f::Float64,
                      inner_f::Float64,
                      outer_p::Float64,
                      inner_p::Float64)

    outer_r = r  * outer_f
    inner_r = r  * inner_f

    # half-life is 1/6 outer - inner
    hl = 3.0 * ln_hlf / abs(outer_r - inner_r)

    xlow = clamp_and_round(x0 - outer_r, w)
    xhigh = clamp_and_round(x0 + outer_r, w)
    ylow = clamp_and_round(y0 - outer_r, h)
    yhigh = clamp_and_round(y0 + outer_r, h)
    n = (xhigh - xlow + 1) * (yhigh - ylow + 1)
    Is = zeros(Int64, n)
    Js = zeros(Int64, n)
    Vs = zeros(Float64, n)
    k = 0
    for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        k +=1
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        # flip i and j in mask
        Is[k] = j
        Js[k] = i
        (dst > outer_r) && continue
        Vs[k] = (dst <= inner_r ) ? inner_p : outer_p * exp(hl * dst)
    end
    sparse(Is, Js, Vs, h, w)
end
# function exp_dot_mask(x0::Float64, y0::Float64,
#                       r::Float64,
#                       w::Int64, h::Int64,
#                       outer_f::Float64,
#                       inner_f::Float64,
#                       outer_p::Float64,
#                       inner_p::Float64)

#     outer_r = r  * outer_f
#     inner_r = r  * inner_f

#     # half-life is 1/6 outer - inner
#     hl = 3.0 * ln_hlf / abs(outer_r - inner_r)

#     xlow = clamp_and_round(x0 - outer_r, w)
#     xhigh = clamp_and_round(x0 + outer_r, w)
#     ylow = clamp_and_round(y0 - outer_r, h)
#     yhigh = clamp_and_round(y0 + outer_r, h)
#     Is = Int64[]
#     Js = Int64[]
#     Vs = Float64[]
#     for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
#         dst = sqrt((i - x0)^2 + (j - y0)^2)
#         (dst > outer_r) && continue
#         v = (dst <= inner_r ) ? inner_p : outer_p * exp(hl * dst)
#         # flip i and j in mask
#         push!(Is, j)
#         push!(Js, i)
#         push!(Vs, v)
#     end
#     sparse(Is, Js, Vs, h, w)
# end

function triangular_dot_mask(x0::Float64, y0::Float64,
                             r::Float64,
                             w::Int64, h::Int64,
                             outer_f::Float64,
                             inner_f::Float64,
                             outer_p::Float64,
                             inner_p::Float64)

    outer_r = r  * outer_f
    inner_r = r  * inner_f

    slope = outer_p  / abs(outer_r - inner_r)

    xlow = clamp_and_round(x0 - outer_r, w)
    xhigh = clamp_and_round(x0 + outer_r, w)
    ylow = clamp_and_round(y0 - outer_r, h)
    yhigh = clamp_and_round(y0 + outer_r, h)
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]
    for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        (dst > outer_r) && continue
        v = (dst <= inner_r ) ? inner_p : clamp(outer_p - slope * dst, 0., 1.0)
        # flip i and j in mask
        push!(Is, j)
        push!(Js, i)
        push!(Vs, v)
    end
    sparse(Is, Js, Vs, h, w)

end

function mixture_dot_mask(x0::Float64, y0::Float64,
                          r::Float64, w::Int64, h::Int64,
                          outer_f::Float64,
                          inner_f::Float64,
                          outer_p::Float64,
                          inner_p::Float64)
    outer_r = r  * outer_f
    inner_r = r  * inner_f

    xlow = clamp_and_round(x0 - outer_r, w)
    xhigh = clamp_and_round(x0 + outer_r, w)
    ylow = clamp_and_round(y0 - outer_r, h)
    yhigh = clamp_and_round(y0 + outer_r, h)
    Is = Int64[]
    Js = Int64[]
    Vs = Float64[]
    for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        (dst > outer_r) && continue
        v = (dst <= inner_r ) ? inner_p : outer_p
        # flip i and j in mask
        push!(Is, j)
        push!(Js, i)
        push!(Vs, v)
    end
    sparse(Is, Js, Vs, h, w)
end


"""
    get_bit_masks(cgs::Vector{CausalGraph},
                       graphics::AbstractGraphics,
                       gm::AbstractGMParams;
                       background=false)

    Returns a Vector{Vector{BitMatrix}} with masks according to
    the Vector{CausalGraph} descrbing the scene

...
# Arguments:
- cgs::Vector{CausalGraph} : causal graphs describing the scene
- graphics : graphical parameters
- gm : generative model parameters
- background : true if you want background masks (i.e.
               1s where there are no objects)
"""
function get_bit_masks(cgs::Vector{CausalGraph},
                       graphics::AbstractGraphics,
                       gm::AbstractGMParams;
                       background=false)

    k = length(cgs)
    masks = Vector{Vector{BitMatrix}}(undef, k)
    
    for t=1:k
        @debug "get_masks timestep: $t / $k \r"
        masks[t] = get_bit_masks(cgs[t], graphics, gm)
    end

    return masks
end

