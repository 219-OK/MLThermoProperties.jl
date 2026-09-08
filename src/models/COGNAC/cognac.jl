include("utils.jl")
include("layers_cognac.jl")

abstract type COGNACfullModel <: CL.UNIFACModel end
abstract type COGNACModel <: CL.ogUNIFACModel end

# reference temperature of the network's reduced interaction basis
const COGNAC_T_REF = 298.15

reduced_to_raw_abc(A, B, C) = (
    A .- B .+ C,
    (B .- 2 .* C) ./ COGNAC_T_REF, 
    C ./ COGNAC_T_REF^2
)

include("cognac_exp.jl")
include("cognac_imp.jl")
include("cognac_lite.jl")

# helper functions for full cognac
CL.default_locations(::Type{M}) where {M<:COGNACfullModel} = ["properties/identifiers.csv"]
get_model_path(::Type{M}) where {M<:COGNACfullModel} = joinpath(DB_PATH, string(nameof(M)))

# constructor functions for full cognac
function _cognac_full(::Type{M}, components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
) where {M<:COGNACfullModel}
    _components = CL.format_components(components)
    _params = CL.getparams(_components, CL.default_locations(M);
        userlocations, ignore_headers = ["dipprnumber", "inchikey", "cas", "canonicalsmiles"])
    smiles = _params["SMILES"].values

    ps, st = load(joinpath(get_model_path(M), "parameters_states.jld2"), "ps", "st")
    nn = COGNACLux()

    graph, molecule_index = cognac_gnngraph(smiles)
    (R, Q, A, B, C), _ = nn(graph, ps, Lux.testmode(st))

    group_names = cognac_group_names(graph, molecule_index)
    groups = cognac_groups(_components, group_names, molecule_index, nameof(M))

    a, b, c = reduced_to_raw_abc(Float64.(A), Float64.(B), Float64.(C))
    params = CL.UNIFACParam(
        PairParam("A", group_names, a),
        PairParam("B", group_names, b),
        PairParam("C", group_names, c),
        SingleParam("R", group_names, Float64.(R)),
        SingleParam("Q", group_names, Float64.(Q)),
    )

    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String[]
    model = M(_components, groups, params, _puremodel, references,
        CL.UNIFACCache(groups, params))
    CL.set_reference_state!(model, reference_state; verbose)

    return model
end

export COGNACexp, COGNACimp, COGNAClite