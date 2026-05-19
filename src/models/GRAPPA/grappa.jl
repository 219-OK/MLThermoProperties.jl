include("utils.jl")
include("layers_grappa.jl")
include("grappa_python.jl")

abstract type GRAPPAModel{T} <: CL.SaturationModel end

struct GRAPPAParam{T} <: CL.ParametricEoSParam{T}
    Tc::CL.SingleParam{T}
    A::CL.SingleParam{T}
    B::CL.SingleParam{T}
    C::CL.SingleParam{T}
end

struct GRAPPA{T} <: GRAPPAModel{T}
    components::Array{String, 1}
    params::GRAPPAParam{T}
    references::Array{String, 1}
end

"""
    GRAPPA{T} <: SaturationModel
    
    GRAPPA(
        components;
        userlocations = String[],
        verbose::Bool=false
    )

## Description

GRAPPA model for calculating vapor pressure of pure components based on the Antoine equation.
On model construction, the Antoine parameters are predicted using a Python implementation the GRAPPA model.

For predicting the Antoine parameters, only the smiles of the molecule is required.
It will automatically be retrieved from the `Clapeyron.jl` database.
The smiles can also be provided by the `userlocations` keyword (see example below). 

## Example

```julia
using Clapeyron, PythonCall

model = GRAPPA("propanol")
model = GRAPPA("propanol"; userlocations=(; smiles="CCCO"))

ps, _, _ = saturation_pressure(model, 300.)         # Vapor pressure at 300 K
```

## References

1.  M. Hoffmann, H. Hasse, and F. Jirasek: GRAPPA—A Hybrid Graph Neural Network for Predicting Pure Component Vapor Pressures, Chemical Engineering Journal Advances 22 (2025) 100750, DOI: https://doi.org/10.1016/j.ceja.2025.100750.

"""
GRAPPA

CL.default_locations(::Type{GRAPPA}) = ["properties/critical.csv", "properties/identifiers.csv"]
get_model_path(::Type{GRAPPA}) = joinpath(DB_PATH, "GRAPPA")

# GRAPPA
function GRAPPA(components; userlocations=String[], reference_state=nothing, verbose=false)
    components = CL.format_components(components)
    _params = CL.getparams(
        components, 
        CL.default_locations(GRAPPA); 
        userlocations, 
        ignore_missing_singleparams = ["Tc",],
        ignore_headers=["dipprnumber","inchikey","cas","canonicalsmiles","Pc","Vc","acentricfactor"]
    )
    
    # load parameters and model
    ps, st = load(joinpath(get_model_path(GRAPPA), "parameters_states_all_grappa.jld2"), "ps", "st")
    st = Lux.testmode(st) 
    grappa_model = GRAPPAModelLux() 

    # get antoine parameters
    _ABC = [
        Float64.(first(grappa_model(smiles_to_gnngraph(s), ps, st)))
    for s in _params["SMILES"].values]
    
    A = CL.SingleParam("A", components, [_abc[1] for _abc in _ABC])
    B = CL.SingleParam("B", components, [_abc[2] for _abc in _ABC])
    C = CL.SingleParam("C", components, [_abc[3] for _abc in _ABC])
    _T = Base.promote_eltype(A,B,C)
    
    params = GRAPPAParam(_params["Tc"],A,B,C)
    references = ["10.1016/j.ceja.2025.100750"]

    return GRAPPA(components, params, references)
end

# vapor pressure calculation
function CL.crit_pure(model::GRAPPAModel{_T}) where _T
    CL.single_component_check(crit_pure, model)
    if only(model.params.Tc.ismissingvalues)
        nan = zero(_T)/zero(_T)
        return nan, nan, nan
    else
        Tc = only(model.params.Tc.values)
    end
    Pc, _, _ = saturation_pressure(model, Tc)
    return (Tc, Pc, NaN)
end

function CL.saturation_pressure_impl(model::GRAPPAModel, T, ::CL.SaturationCorrelation)
    nan = zero(T)/zero(T)
    Tc = only(model.params.Tc.ismissingvalues) ? nan : only(model.params.Tc.values)
    A = only(model.params.A.values)
    B = only(model.params.B.values)
    C = only(model.params.C.values)

    !isnan(Tc) && T > Tc && (return nan, nan, nan)
    psat = exp(A - B/(T + C)) * 1000
    return psat, nan, nan
end

export GRAPPA
