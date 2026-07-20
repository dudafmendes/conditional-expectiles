module GARCHModels

using Distributions
using Optim
using Random
using ForwardDiff
using ADTypes
using LinearAlgebra
using Statistics

export GARCHModel,
       simulate,
       simulate_next_observation,
       forecast_variance,
       residuals,
       garch_variance,
       garch_variance_fixed,
       garch_variance_fixed!,
       garch_design_matrix,
       garch_design_matrix!,
       garch_negloglik,
       garch_likelihood_fixed!,
       garch_parameter_variance,
       estimate,
       estimate_fixed,
       data_from_residuals_variance

include("types.jl")
include("recursion.jl")
include("simulation.jl")
include("likelihood.jl")
include("estimation.jl")
include("derivatives.jl")

end
