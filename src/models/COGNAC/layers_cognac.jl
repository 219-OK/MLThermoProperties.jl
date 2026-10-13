## COGNAC networks: atom/bond encoders, bonded-graph message passing, R/Q and interaction heads

# reference cavity volume/area the checkpoint's R/Q normalization was trained with
const COGNAC_V_REF = 20.18
const COGNAC_A_REF = 21.17

const COGNAC_ATOM_VOCAB = (119, 8, 9, 6, 2, 6)  # Z, degree, charge, hybridization, aromatic, nH
const COGNAC_BOND_VOCAB = (5, 2, 2)             # bond type, conjugated, in ring

# sum of categorical embeddings + LayerNorm (atom and bond encoder differ only in vocabulary)
function categorical_encoder(vocab, hidden_dim)
    return Chain(
        Parallel(+, map(n -> Embedding(n => hidden_dim), vocab)...),
        LayerNorm((hidden_dim,), identity; dims = 1),
    )
end

# 1-based indices per categorical feature row, as `Parallel` expects them
encoder_input(features::AbstractMatrix{<:Integer}) =
    ntuple(k -> view(features, k, :) .+ 1, size(features, 1))

# MessagePassing over bonded (intra-molecule) edges only
struct LocalMessagePassingLux <: LuxCore.AbstractLuxContainerLayer{(:msg, :upd, :norm)}
    msg::Chain
    upd::Chain
    norm::LayerNorm
end

function LocalMessagePassingLux(hidden_dim::Int)
    return LocalMessagePassingLux(
        Chain(Dense(3hidden_dim => hidden_dim, swish), Dense(hidden_dim => hidden_dim)),
        Chain(Dense(2hidden_dim => hidden_dim, swish), Dense(hidden_dim => hidden_dim)),
        LayerNorm((hidden_dim,), identity; dims = 1),
    )
end

function (layer::LocalMessagePassingLux)(graph::GNNGraph, h, e, ps, st::NamedTuple)
    if graph.num_edges > 0
        msg = StatefulLuxLayer{true}(layer.msg, ps.msg, st.msg)
        m = GNNlib.propagate((xi, xj, ee) -> msg(vcat(xi, xj, ee)), graph, +; xi = h, xj = h, e)
        upd = StatefulLuxLayer{true}(layer.upd, ps.upd, st.upd)
        h = h .+ upd(vcat(h, m))
    end
    return first(layer.norm(h, ps.norm, st.norm)), st
end

struct LocalMPNNLux <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    layers::Tuple
end

LocalMPNNLux(hidden_dim::Int, num_layers::Int) =
    LocalMPNNLux(ntuple(_ -> LocalMessagePassingLux(hidden_dim), num_layers))

function (model::LocalMPNNLux)(graph::GNNGraph, h, e, ps, st::NamedTuple)
    for (i, k) in enumerate(keys(ps.layers))
        h, _ = model.layers[i](graph, h, e, ps.layers[k], st.layers[k])
    end
    return h, st
end

# per-atom embeddings after message passing
function atom_embeddings(model, graph::GNNGraph, ps, st::NamedTuple)
    h, _ = model.atom_encoder(encoder_input(graph.ndata.x), ps.atom_encoder, st.atom_encoder)
    e, _ = model.edge_encoder(encoder_input(graph.edata.e), ps.edge_encoder, st.edge_encoder)
    return first(model.gnn(graph, h, e, ps.gnn, st.gnn))
end

struct COGNACLux{T} <:
       LuxCore.AbstractLuxContainerLayer{(:atom_encoder, :edge_encoder, :gnn, :radius_volume_head, :interaction_head)}
    atom_encoder::Chain
    edge_encoder::Chain
    gnn::LocalMPNNLux
    radius_volume_head::Chain
    interaction_head::Chain
    eps::T
end

function COGNACLux(; hidden_dim = 256, num_layers = 4, eps = 1.0f-8)
    return COGNACLux(
        categorical_encoder(COGNAC_ATOM_VOCAB, hidden_dim),
        categorical_encoder(COGNAC_BOND_VOCAB, hidden_dim),
        LocalMPNNLux(hidden_dim, num_layers),
        Chain(
            Dense(hidden_dim => hidden_dim, swish),
            Dense(hidden_dim => hidden_dim ÷ 2, swish),
            Dense(hidden_dim ÷ 2 => 2),
        ),
        Chain(
            Dense(3hidden_dim => hidden_dim, swish),
            Dense(hidden_dim => hidden_dim ÷ 2, swish),
            Dense(hidden_dim ÷ 2 => 3),
        ),
        eps,
    )
end

# c_ab → c_ab - (c_aa + c_bb)/2: exchange coefficients, zero on the diagonal, so Ψ_aa == 1
function exchange_subtract(M::AbstractMatrix)
    d = diag(M)
    return M .- (d .+ d') ./ 2
end

function (model::COGNACLux)(graph::GNNGraph, ps, st::NamedTuple)
    h = atom_embeddings(model, graph, ps, st)

    raw, _ = model.radius_volume_head(h, ps.radius_volume_head, st.radius_volume_head)
    R = (NNlib.softplus.(view(raw, 1, :)) .+ model.eps) ./ COGNAC_V_REF
    Q = (NNlib.softplus.(view(raw, 2, :)) .+ model.eps) ./ COGNAC_A_REF

    n = size(h, 2)
    ia, ib = repeat(1:n, outer = n), repeat(1:n, inner = n)
    ha, hb = h[:, ia], h[:, ib]
    pair, _ = model.interaction_head(
        vcat(ha .+ hb, abs.(ha .- hb), ha .* hb), ps.interaction_head, st.interaction_head
    )
    # (A, B, C) of a_eff(T) = A + Bτ + Cτ², τ = T/T_ref - 1
    A, B, C = ntuple(k -> exchange_subtract(reshape(pair[k, :], n, n)), 3)

    return (R, Q, A, B, C), st
end
