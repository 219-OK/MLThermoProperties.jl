include("utils.jl")
include("layers_cognac.jl")

abstract type COGNACModel <: CL.UNIFACModel end

# reference temperature of the network's reduced interaction basis
const COGNAC_T_REF = 298.15

reduced_to_raw_abc(A, B, C) = (
    A .- B .+ C,
    (B .- 2 .* C) ./ COGNAC_T_REF,
    C ./ COGNAC_T_REF^2
)

include("cognac_exp.jl")
include("cognac_imp.jl")

# helper functions
CL.default_locations(::Type{M}) where {M<:COGNACModel} = ["properties/identifiers.csv"]
get_model_path(::Type{M}) where {M<:COGNACModel} = joinpath(DB_PATH, string(nameof(M)))

# constructor function shared by both variants
function _cognac(::Type{M}, components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
) where {M<:COGNACModel}
    _components = CL.format_components(components)
    _params = CL.getparams(_components, CL.default_locations(M);
        userlocations, ignore_headers = ["dipprnumber", "inchikey", "cas", "canonicalsmiles"])
    smiles = _params["SMILES"].values

    ps, st = load(joinpath(get_model_path(M), "parameters_states.jld2"), "ps", "st")
    nn = COGNACLux()

    graph, molecule_index, classes = cognac_gnngraph(smiles)
    (R, Q, A, B, C), _ = nn(graph, ps, Lux.testmode(st))

    # symmetry-equivalent atoms of a molecule are one group, counted by ν
    R_groups, Q_groups, (A_groups, B_groups, C_groups) = cognac_group_params(
        classes, Float64.(R), Float64.(Q), map(x -> Float64.(x), (A, B, C)))

    group_names = cognac_group_names(graph, molecule_index, classes)
    groups = cognac_groups(_components, group_names, molecule_index, classes, nameof(M))

    a, b, c = reduced_to_raw_abc(A_groups, B_groups, C_groups)
    params = CL.UNIFACParam(
        PairParam("A", group_names, a),
        PairParam("B", group_names, b),
        PairParam("C", group_names, c),
        SingleParam("R", group_names, R_groups),
        SingleParam("Q", group_names, Q_groups),
    )

    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String[]
    model = M(_components, groups, params, _puremodel, references,
        CL.UNIFACCache(groups, params))
    CL.set_reference_state!(model, reference_state; verbose)

    return model
end

# lnΓₖ/Qₖ = 1 - log(sₖ) - ∑ₘ Ψₖₘ Θₘ/sₘ with sₖ = ∑ₘ Θₘ Ψₘₖ, `s` being scratch
function cognac_g_explicit!(g, s, Θ, Ψ)
    mul!(s, transpose(Ψ), Θ)
    g .= 1 .- log.(s)
    s .= Θ ./ s
    mul!(g, Ψ, s, -1, 1)
    return g
end

# lnγ of both variants, which differ only in `cognac_g!`
function CL.lnγ_impl!(lnγ, model::COGNACModel, p, T, z)
    ν = model.groups.n_flattenedgroups
    Q = model.params.Q.values
    # plain mod. UNIFAC Ψ: the model's soft saturation above |a/T| = 40 never triggers for
    # the shipped checkpoints, whose worst value is 15
    Ψ = CL.Ψ(model, p, T, z)
    TT = Base.promote_eltype(Ψ, z)
    nc, ng = length(z), length(Q)

    # group environments: the mixture, and one pure-component reference per molecule,
    # defined on that molecule's own groups alone (`live`)
    Θ = zeros(TT, ng)
    Θᵢ = zeros(TT, ng, nc)
    live = falses(ng, nc)
    for i in 1:nc
        νᵢ = ν[i]
        w = 1 / dot(Q, νᵢ)
        for k in 1:ng
            iszero(νᵢ[k]) && continue
            Θ[k] += Q[k] * νᵢ[k] * z[i]
            Θᵢ[k, i] = Q[k] * νᵢ[k] * w
            live[k, i] = true
        end
    end
    Θ ./= sum(Θ)

    g = cognac_g!(model, similar(Θ), Θ, Ψ, nothing)
    gᵢ = cognac_g!(model, similar(Θᵢ), Θᵢ, Ψ, live)

    r, q, q_p = model.unifac_cache.r, model.unifac_cache.q, model.unifac_cache.q_p
    ∑r, ∑q, ∑q_p, n = dot(z, r), dot(z, q), dot(z, q_p), sum(z)
    for i in 1:nc
        V′ = q_p[i] * n / ∑q_p       # the Dortmund r^(3/4) modification
        VF = r[i] * ∑q / (∑r * q[i])
        νᵢ = ν[i]
        residual = zero(TT)
        for k in 1:ng
            iszero(νᵢ[k]) && continue
            residual += νᵢ[k] * Q[k] * (g[k] - gᵢ[k, i])
        end
        # z/2 = 5 with UNIFAC's coordination number z = 10
        lnγ[i] = 1 - V′ + log(V′) - 5 * q[i] * (1 - VF + log(VF)) + residual
    end
    return lnγ
end

# `CL.UNIFACModel`'s own gᴱ would evaluate the explicit closure, not COGNACimp's
CL.excess_gibbs_free_energy(model::COGNACModel, p, T, z) =
    CL.Rgas(model) * T * dot(z, CL.lnγ(model, p, T, z))

# Clapeyron's generic ∂lnγ/∂T calls `lnγ_impl!` with the temperature alone, which errors
CL.∂lnγ∂T(model::COGNACModel, p, T, z, cache = nothing) =
    CL.Solvers.derivative(t -> CL.lnγ(model, p, t, z), T)

const COGNAC = COGNACexp

export COGNAC, COGNACexp, COGNACimp
