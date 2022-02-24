abstract type GenerativeModel end
abstract type GMState end
abstract type Thing end


"""
Evolves state according rules in generative model
"""
function step(::GenerativeModel, ::GMState)::GMState end

function render(::GenerativeModel, ::GMState) end

include("repulsion/repulsion.jl")
