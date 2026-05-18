@concrete struct GATv2Conv <: AbstractLuxLayer
    dense_i
    dense_j
    dense_e
    init_weight
    init_bias
    use_bias::Bool
    σ
    negative_slope
    channel::Pair{NTuple{2, Int}, Int}
    heads::Int
    concat::Bool
    add_self_loops::Bool
    dropout
end

function GATv2Conv(ch::Pair{Int, Int}, args...; kws...)
    GATv2Conv((ch[1], 0) => ch[2], args...; kws...)
end

function GATv2Conv(ch::Pair{NTuple{2, Int}, Int},
                   σ = identity;
                   heads::Int = 1,
                   concat::Bool = true,
                   negative_slope = 0.2,
                   init_weight = glorot_uniform,
                   init_bias = zeros32,
                   use_bias::Bool = true,
                   add_self_loops = true,
                   dropout=0.0)

    (in, ein), out = ch

    if add_self_loops
        @assert ein==0 "Using edge features and setting add_self_loops=true at the same time is not yet supported."
    end

    dense_i = Dense(in => out * heads; use_bias, init_weight, init_bias)
    dense_j = Dense(in => out * heads; use_bias = false, init_weight)
    if ein > 0
        dense_e = Dense(ein => out * heads; use_bias = false, init_weight)
    else
        dense_e = nothing
    end
    return GATv2Conv(dense_i, dense_j, dense_e, 
                     init_weight, init_bias, use_bias, 
                    σ, negative_slope, 
                    ch, heads, concat, add_self_loops, dropout)
end


LuxCore.outputsize(l::GATv2Conv) = (l.concat ? l.channel[2]*l.heads : l.channel[2],)
##TODO: parameterlength

function LuxCore.initialparameters(rng::AbstractRNG, l::GATv2Conv)
    (in, ein), out = l.channel
    dense_i = LuxCore.initialparameters(rng, l.dense_i)
    dense_j = LuxCore.initialparameters(rng, l.dense_j)
    a = l.init_weight(out, l.heads)
    ps = (; dense_i, dense_j, a)
    if ein > 0
        ps = (ps..., dense_e = LuxCore.initialparameters(rng, l.dense_e))
    end
    if l.use_bias
        ps = (ps..., bias = l.init_bias(rng, l.concat ? out * l.heads : out))
    end
    return ps
end

(l::GATv2Conv)(g, x, ps, st) = l(g, x, nothing, ps, st)

function (l::GATv2Conv)(g, x, e, ps, st)
    dense_i = StatefulLuxLayer{true}(l.dense_i, ps.dense_i, _getstate(st, :dense_i))
    dense_j = StatefulLuxLayer{true}(l.dense_j, ps.dense_j, _getstate(st, :dense_j))
    dense_e = l.dense_e === nothing ? nothing : 
              StatefulLuxLayer{true}(l.dense_e, ps.dense_e, _getstate(st, :dense_e))

    m = (; l.add_self_loops, l.channel, l.heads, l.concat, l.dropout, l.σ, 
           ps.a, bias = _getbias(ps), dense_i, dense_j, dense_e, l.negative_slope)
    return GNNlib.gatv2_conv(m, g, x, e), st
end

function Base.show(io::IO, l::GATv2Conv)
    (in, ein), out = l.channel
    print(io, "GATv2Conv(", ein == 0 ? in : (in, ein), " => ", out ÷ l.heads)
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ", negative_slope=", l.negative_slope)
    print(io, ")")
end

# getstate
_getbias(ps) = hasproperty(ps, :bias) ? getproperty(ps, :bias) : false
_getstate(st, name) = hasproperty(st, name) ? getproperty(st, name) : NamedTuple()
_getstate(s::StatefulLuxLayer{Val{true}}) = s.st
_getstate(s::StatefulLuxLayer{Val{false}}) = s.st_any