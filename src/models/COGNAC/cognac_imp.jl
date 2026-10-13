struct COGNACimp{c<:CL.EoSModel,T} <: COGNACModel
    components::Array{String,1}
    groups::GroupParam{T}
    params::CL.UNIFACParam{T}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
    unifac_cache::CL.UNIFACCache{T}
end

function COGNACimp(components, groups, params, puremodel, references, unifac_cache)
    c = eltype(puremodel)
    T = eltype(params)
    return COGNACimp{c,T}(components, groups, params, puremodel, references, unifac_cache)
end

COGNACimp(components; kwargs...) = _cognac(COGNACimp, components; kwargs...)

"""
    COGNACimp <: UNIFACModel

    COGNACimp(components;
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
predicting activity coefficients, implicit version.
Identical to [`COGNACexp`](@ref) up to the residual term: the group activity coefficients
are the solution of the COSMOSPACE self-consistency condition

```
lnΓₖ/Qₖ = -log(∑ₘ Θₘ exp(lnΓₘ/Qₘ) Ψₘₖ)
```

which is solved by damped fixed-point iteration, started from the explicit closure (the two
are nested). This is more accurate at infinite dilution, but the parameter table alone is
not enough to reproduce the model in a stock mod. UNIFAC implementation.

## Example
```julia
using MLThermoProperties, Clapeyron

model = COGNACimp(["ethanol", "water"])
model = COGNACimp(["ethanol", "water"]; userlocations=(; SMILES=["CCO", "O"]))

activity_coefficient(model, 1e5, 298.15, [0.3, 0.7])
```

## References
1. Hoffmann et al., to be published.
"""
COGNACimp

const COGNAC_SOLVER = (damping = 0.7, tol = 5.0e-3, max_iter = 25, extra_steps = 10)

# one damped application of the COSMOSPACE map, returning max|Δg| over the live groups
function cosmospace_step!(g, w, rhs, Θ, Ψ, live)
    α = COGNAC_SOLVER.damping
    w .= Θ .* exp.(g)
    mul!(rhs, transpose(Ψ), w)
    Δ = zero(eltype(g))
    for k in eachindex(g)
        (live === nothing || live[k]) || continue
        gₖ = (1 - α) * g[k] - α * log(rhs[k])
        Δ = max(Δ, abs(gₖ - g[k]))
        g[k] = gₖ
    end
    return Δ
end

# lnΓₖ/Qₖ as the COSMOSPACE fixed point, warm-started at the nested explicit closure
function cognac_g!(::COGNACimp, g, Θ, Ψ, live)
    w, rhs = similar(Θ), similar(Θ)
    cognac_g_explicit!(g, w, Θ, Ψ)
    live === nothing || (g .= ifelse.(live, g, zero(eltype(g))))

    for _ in 1:COGNAC_SOLVER.max_iter
        cosmospace_step!(g, w, rhs, Θ, Ψ, live) ≤ COGNAC_SOLVER.tol && break
    end
    for _ in 1:COGNAC_SOLVER.extra_steps
        cosmospace_step!(g, w, rhs, Θ, Ψ, live)
    end
    return g
end
