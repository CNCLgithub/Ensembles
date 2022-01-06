export NullGraphics
################################################################################
# Null Graphics
################################################################################

struct NullGraphics <: AbstractGraphics end

function render(::NullGraphics, ::CausalGraph)
    Diff()
end
function predict(::NullGraphics, ::CausalGraph)
    error("not implemented")
end
