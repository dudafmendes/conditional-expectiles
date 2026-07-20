

"""
    GARCHModel{D<:Distribution}

A structure representing a Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) model.

# Fields
- `ω::Float64`: Intercept term (constant volatility component)
- `α::Vector{Float64}`: ARCH coefficients (squared innovation terms)
- `β::Vector{Float64}`: GARCH coefficients (lagged variance terms)
- `γ::Union{Float64, Nothing}`: Optional leverage parameter for asymmetric effects (leverage term)
- `dist::D`: Instrumental distribution for QML (e.g., Normal, Student's t)
"""
struct GARCHModel{D<:Distribution}
    ω::Float64
    α::Vector{Float64}
    β::Vector{Float64}
    γ::Union{Float64, Nothing}
    dist::D
end

"""
    Base.convert(::Type{Vector{Float64}}, model::GARCHModel) -> Vector{Float64}

Convert a GARCHModel to a vector containing all model parameters in sequential order.

# Arguments
- `model::GARCHModel`: The GARCH model to convert

# Returns
- `Vector{Float64}`: A vector containing [ω, α..., β..., γ] where γ is included only if not nothing
"""
function Base.convert(::Type{Vector{Float64}}, model::GARCHModel)
    params = Float64[model.ω]
    append!(params, model.α)
    append!(params, model.β)
    if model.γ !== nothing
        push!(params, model.γ)
    end
    return params
end

"""
    Base.show(io::IO, model::GARCHModel)

Display the GARCHModel in a formatted, readable manner.

# Arguments
- `io::IO`: The output stream to write to
- `model::GARCHModel`: The GARCH model to display
"""
function Base.show(io::IO, model::GARCHModel)
    println(io, "GARCHModel")
    println(io, "  ω = ", model.ω)
    println(io, "  α = ", model.α)
    println(io, "  β = ", model.β)
    println(io, "  γ = ", isnothing(model.γ) ? "nothing" : model.γ)
    println(io, "  dist = ", model.dist)
end
