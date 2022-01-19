module Ensembles

using Gen

# defines generative models that can be used as data generating procedures
include("generative_models/generative_models.jl")

# interface for defining dgps
# defines things like datasets
#include("dgp/dgp.jl")

# TODO: implement me!

# include("implicit_gms/implicit_gms.jl")

# include("flows/flows.jl")

function greet()
	print("Hello there")
end

end # module
