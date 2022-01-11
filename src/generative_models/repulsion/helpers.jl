
const two_pi_sqr = 4.0 * pi * pi

# translates coordinate from euclidean to image space
function translate_area_to_img(x::Float64, y::Float64,
                               img_width::Int64, img_height::Int64,
                               area_width::Float64, area_height::Float64)

    x *= img_width/area_width
    x += img_width/2

    # inverting y
    y *= -1 * img_height/area_height
    y += img_height/2

    return x, y
end

function clamp_and_round(v::Float64, c::Int64)::Int64
    @> v begin
        clamp(1., c)
        (@>> round(Int64))
    end
end

"""
Writes out a dot mask to matrix
"""
function exp_dot_mask!(g::Matrix{Float64},
                       x0::Float64, y0::Float64,
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
    for (i, j) in Iterators.product(xlow:xhigh, ylow:yhigh)
        k +=1
        dst = sqrt((i - x0)^2 + (j - y0)^2)
        (dst > outer_r) && continue
        # flip i and j in mask
        g[j, i] = (dst <= inner_r ) ? inner_p : outer_p * exp(hl * dst)
    end
    return nothing
end

# initializes a new dot
function Dot(gm::RepulsionGM,
             pos::SVector{2, Float64}, vel::SVector{2, Float64})
    Dot(gm.dot_radius, pos, vel, spzeros(gm.dimensions))
end
