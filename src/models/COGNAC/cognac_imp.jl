struct COGNACimp{c<:CL.EoSModel,T} <: COGNACfullModel
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

COGNACimp(components; kwargs...) = _cognac_full(COGNACimp, components; kwargs...)

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

which is solved by damped fixed-point iteration with the settings the checkpoint was
trained with. This is more accurate at infinite dilution, but the parameter table alone is
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

# solver settings
const COGNAC_SOLVER = (damping = 0.7, tol = 5.0e-3, max_iter = 25, extra_steps = 10)

# one damped application of the COSMOSPACE map, with the largest update it made on `live`
function cosmospace_step(g, Θ, Ψ, live)
    α = COGNAC_SOLVER.damping
    rhs = .-log.(Ψ' * (Θ .* exp.(g)))
    gnew = ifelse.(live, (1 - α) .* g .+ α .* rhs, zero(eltype(g)))
    return gnew, CL.Solvers.dnorm(vec(gnew), vec(g), Inf)
end

# iterative cosmospace solver
function cosmospace_g(Θ, Ψ, live)
    s = Ψ' * Θ
    g = ifelse.(live, 1 .- log.(s) .- Ψ * (Θ ./ s), zero(eltype(s)))
    for _ in 1:COGNAC_SOLVER.max_iter
        g, Δ = cosmospace_step(g, Θ, Ψ, live)
        Δ ≤ COGNAC_SOLVER.tol && break
    end
    for _ in 1:COGNAC_SOLVER.extra_steps
        g, _ = cosmospace_step(g, Θ, Ψ, live)
    end
    return g
end

function CL.lnγ_impl!(lnγ, model::COGNACimp, p, T, z)
    x = z ./ sum(z)
    Q = model.params.Q.values
    ν = model.groups.n_flattenedgroups
    Ψ = CL.Ψ(model, p, T, z)

    # mixture environment, and one column per pure-component reference
    Θ = Q .* sum(x[i] .* ν[i] for i in eachindex(x))
    Θ ./= sum(Θ)
    Θᵢ = reduce(hcat, (Q .* ν[i]) ./ dot(Q, ν[i]) for i in eachindex(x))
    liveᵢ = reduce(hcat, ν) .> 0

    lnΓ = Q .* cosmospace_g(Θ, Ψ, trues(length(Q)))
    lnΓᵢ = Q .* cosmospace_g(Θᵢ, Ψ, liveᵢ)

    r, q, q_p = model.unifac_cache.r, model.unifac_cache.q, model.unifac_cache.q_p
    V = r ./ dot(x, r)
    V′ = q_p ./ dot(x, q_p)     # the Dortmund r^(3/4) modification
    F = q ./ dot(x, q)
    for i in eachindex(x)
        # z/2 = 5 with UNIFAC's coordination number z = 10
        lnγ[i] = 1 - V′[i] + log(V′[i]) - 5 * q[i] * (1 - V[i] / F[i] + log(V[i] / F[i]))
        lnγ[i] += dot(ν[i], lnΓ) - dot(ν[i], view(lnΓᵢ, :, i))
    end
    return lnγ
end

function CL.excess_gibbs_free_energy(model::COGNACimp, p, T, z) 
    return CL.Rgas(model) * T * dot(z, CL.lnγ(model, p, T, z))
end
