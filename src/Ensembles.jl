module Ensembles

using Gen
using Lazy
using Parameters
using SparseArrays
using StaticArrays
using Distributions

include("utils/utils.jl")

# defines generative models that can be used as data generating procedures
include("generative_models/generative_models.jl")

# interface for defining dgps
# defines things like datasets
#include("dgp/dgp.jl")

# TODO: implement me!

# include("implicit_gms/implicit_gms.jl")

# include("flows/flows.jl")

@load_generated_functions

end # module
