struct COGNACexp{c<:CL.EoSModel,T} <: COGNACfullModel
    components::Array{String,1}
    groups::GroupParam{T}
    params::CL.UNIFACParam{T}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
    unifac_cache::CL.UNIFACCache{T}
end

function COGNACexp(components, groups, params, puremodel, references, unifac_cache)
    c = eltype(puremodel)
    T = eltype(params)
    return COGNACexp{c,T}(components, groups, params, puremodel, references, unifac_cache)
end

COGNACexp(components; kwargs...) = _cognac_full(COGNACexp, components; kwargs...)

"""
    COGNACexp <: UNIFACModel

    COGNACexp(components;
    puremodel = BasicIdeal,
    userlocations = String[],
    pure_userlocations = String[],
    verbose = false,
    reference_state = nothing)

## Input parameters
- `SMILES`: SMILES representation of the components

## Input models
- `puremodel`: model to calculate pure pressure-dependent properties

## Description
COGNAC (COsmo-Guided Neural Activity Coefficients) model [hoffmann_cognac_2026](@cite) for
predicting activity coefficients, explicit version.
The mod. UNIFAC (Dortmund) parameters `R`, `Q` and `Aᵢⱼ`, `Bᵢⱼ`, `Cᵢⱼ` are predicted per
atom/atom pair based on a graph neural network.
The parameters are only valid for the mixture they were built for; `COGNACexp` therefore
predicts them on model construction.
The group activity coefficients then follow from the closed-form mod. UNIFAC equations.

## Example
```julia
using MLThermoProperties, Clapeyron

model = COGNACexp(["ethanol", "water"])
model = COGNACexp(["ethanol", "water"]; userlocations=(; SMILES=["CCO", "O"]))

activity_coefficient(model, 1e5, 298.15, [0.3, 0.7])
```

## References
1. Hoffmann et al., to be published.
"""
COGNACexp
