struct COGNAClite{c<:CL.EoSModel,T} <: COGNACModel
    components::Array{String,1}
    groups::GroupParam{T}
    params::CL.ogUNIFACParam{T}
    puremodel::CL.EoSVectorParam{c}
    references::Array{String,1}
    unifac_cache::CL.UNIFACCache{T}
end

function COGNAClite(components, groups, params, puremodel, references, unifac_cache)
    c = eltype(puremodel)
    T = eltype(params)
    return COGNAClite{c,T}(components, groups, params, puremodel, references, unifac_cache)
end

"""
    COGNAClite <: ogUNIFACModel

    COGNAClite(components;
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
COGNAC (COsmo-Guided Neural Activity Coefficients) model [hoffmann_cognac_2026](@cite) for predicting acivity coefficients (lite: without implcit solving). 
The UNIFAC parameters `R`, `Q` and `Aᵢⱼ` are predicted per atom/atom pair based on a graph neural network.
The parameters are only valid for the mixture they were built for; `COGNAClite` therefore predicts them on model construction.

## Example
```julia
using MLThermoProperties, Clapeyron

model = COGNAClite(["ethanol", "water"])
model = COGNAClite(["ethanol", "water"]; userlocations=(; SMILES=["CCO", "O"]))

activity_coefficient(model, 1e5, 298.15, [0.3, 0.7])
```

## References
1. Hoffmann et al., to be published.
"""
COGNAClite

CL.default_locations(::Type{COGNAClite}) = ["properties/identifiers.csv"]
get_model_path(::Type{COGNAClite}) = joinpath(DB_PATH, "COGNAClite")

function COGNAClite(components;
        puremodel = BasicIdeal,
        userlocations = String[],
        pure_userlocations = String[],
        verbose = false,
        reference_state = nothing
)
    _components = CL.format_components(components)
    _params = CL.getparams(_components, CL.default_locations(COGNAClite);
        userlocations, ignore_headers = ["dipprnumber", "inchikey", "cas", "canonicalsmiles"])
    smiles = _params["SMILES"].values

    ps, st = load(joinpath(get_model_path(COGNAClite), "parameters_states.jld2"), "ps", "st")
    nn = COGNACliteLux()

    graph, molecule_index = cognac_gnngraph(smiles)
    (R, Q, A), _ = nn(graph, ps, Lux.testmode(st))

    group_names = cognac_group_names(graph, molecule_index)
    groups = cognac_groups(_components, group_names, molecule_index, :COGNAClite)

    params = CL.ogUNIFACParam(
        PairParam("A", group_names, Float64.(A)),
        SingleParam("R", group_names, Float64.(R)),
        SingleParam("Q", group_names, Float64.(Q)),
    )

    _puremodel = CL.init_puremodel(puremodel, components, pure_userlocations, verbose)
    references = String[]
    model = COGNAClite(_components, groups, params, _puremodel, references,
        CL.UNIFACCache(groups, params))
    CL.set_reference_state!(model, reference_state; verbose)

    return model
end