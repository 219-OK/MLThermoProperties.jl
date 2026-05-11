using JLD2
using Lux
using Clapeyron

include("utils.jl")
include("grappa.jl")


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

CL.default_locations(::Type{GRAPPA}) = ["properties/critical.csv", "properties/identifiers.csv"]

# loading grappa weights function
function load_grappa_weights()
    script_dir = @__DIR__ 

    load_parameters = normpath(joinpath(script_dir, "..", "..", "..", "database", "GRAPPA", "parameters_states_all_grappa.jld2"))
    
    parameter_states = load(load_parameters)
    ps = parameter_states["ps"]

    st = Lux.testmode(parameter_states["st"]) 
    
    model = GRAPPAModelLux()
    
    return model, ps, st
end

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
    
    grappa_model, ps, st = load_grappa_weights()
    
    _ABC = Vector{Vector{Float64}}()
    
    for s in _params["SMILES"].values
        graph = smiles_to_gnngraph(s)
    
        antoine_scaled, _ = grappa_model(graph, ps, st)
        
        A_val = Float64(antoine_scaled[1])
        B_val = Float64(antoine_scaled[2])
        C_val = Float64(antoine_scaled[3])
        
        push!(_ABC, [A_val, B_val, C_val])
    end
    
    A = CL.SingleParam("A", components, [_abc[1] for _abc in _ABC])
    B = CL.SingleParam("B", components, [_abc[2] for _abc in _ABC])
    C = CL.SingleParam("C", components, [_abc[3] for _abc in _ABC])
    _T = Base.promote_eltype(A,B,C)
    
    params = GRAPPAParam{_T}(_params["Tc"],A,B,C)

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
