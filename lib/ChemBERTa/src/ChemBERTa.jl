"""
    ChembBERTa

Small sub-package for the ChemBERTa model applied in the MLPROP models.
"""
module ChemBERTa

using DataStructures: OrderedDict
using ConcreteStructs, JSON, Random, SafeTensors

using Lux, NNlib

# Init
const DATADIR = joinpath(pkgdir(@__MODULE__), "data")
rng = Random.default_rng()

include("utils.jl")
include("api.jl")

include("tokenizer/tokenizer.jl")

include("model/bert.jl")
include("model/transformer_encoder.jl")

end # module ChemBERTa
