using NNlib
using GNNGraphs: GNNGraph
using Graphs
using Lux
using LuxCore
using MolecularGraph
using Random
using Statistics
using GNNLux

# GraphAttentionPooling
struct GraphAttentionPoolingLux <: LuxCore.AbstractLuxLayer
    in_dim::Int
    key_dim::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::GraphAttentionPoolingLux)
    return (
        query_weight = Lux.glorot_uniform(rng, layer.key_dim, layer.in_dim) ,
        key_weight   = Lux.glorot_uniform(rng, layer.key_dim, layer.in_dim) ,
        value_weight = Lux.glorot_uniform(rng, layer.in_dim, layer.in_dim) 
    )
end

function LuxCore.initialstates(rng::AbstractRNG, layer::GraphAttentionPoolingLux)
    return NamedTuple()
end


function (layer::GraphAttentionPoolingLux)(node_out::AbstractMatrix, ps, st::NamedTuple)
    # node_out has the form: Features=32, Nodes=N
    n_features = size(node_out, 1)
    
    # calculate matrices
    Q = ps.query_weight' * node_out
    K = ps.key_weight' * node_out
    V = ps.value_weight' * node_out
    
    # calculate attention score (Q^T * K) / sqrt(d)
    attn_logits = (Q' * K) ./ Float32(sqrt(n_features))
    
    attention_scores = softmax(attn_logits, dims=2)
    
    context_matrix = V * attention_scores'
    
    pooled_graph = sum(context_matrix, dims=2)
    
    return pooled_graph, st
end

# MultiHeadAttentionPooling
struct MultiHeadAttentionPoolingLux <: LuxCore.AbstractLuxContainerLayer{(:heads,)}
    heads::Tuple
end

function (layer::MultiHeadAttentionPoolingLux)(node_out, ps, st::NamedTuple)
    head_results = []
    
    for (i,k) in enumerate(keys(ps.heads))
        out, _ = layer.heads[i](node_out, ps.heads[k], st.heads[k])
        push!(head_results, out)
    end
    
    stacked = cat(head_results..., dims=3)
    final_pooled = dropdims(mean(stacked, dims=3), dims=3)
    
    return final_pooled, st
end

# calculate pooling
function MultiHeadAttentionPoolingLux(in_dim::Int, key_dim::Int, num_heads::Int)
    head_layers = Tuple(GraphAttentionPoolingLux(in_dim, key_dim) for _ in 1:num_heads)
    return MultiHeadAttentionPoolingLux(head_layers)
end

"""
build_gnn_gat not used, because GNNChain cannot use edges
"""
# # gnn
# function build_gnn_gat(node_dim=24, edge_dim=9, conv_dim=32, heads=5, num_layers=3)
#     layers = []
    
#     # layer 1: from 24 to 32 dimensions
#     push!(layers, GATv2Conv((node_dim, edge_dim)=> conv_dim; heads=heads, concat=false, add_self_loops=false))
#     push!(layers, elu)
    
#     # layer 2 & 3: stays at 32 dimensions
#     for _ in 1:(num_layers - 1)
#         push!(layers, GATv2Conv((conv_dim, edge_dim) => conv_dim; heads=heads, concat=false, add_self_loops=false))
#         push!(layers, elu)
#     end
    
#     return GNNChain(layers...)
# end



# ---------------------------------------------------------------------------
# prediction head

# input_dim=34, hidden_dim=16, num_hidden_layers=3, out_dim=3
function grappa_head(input_dim=34, hidden_dim=16, num_hidden_layers=3, out_dim=3)
    layers = []
    
    # Input Layer (BatchNorm + Linear + ELU)
    push!(layers, BatchNorm(input_dim))
    push!(layers, Dense(input_dim => hidden_dim, elu))
    
    # Hidden Layers
    for _ in 1:num_hidden_layers
        push!(layers, BatchNorm(hidden_dim))
        push!(layers, Dense(hidden_dim => hidden_dim, elu))
    end
    
    # output layer
    push!(layers, BatchNorm(hidden_dim))
    push!(layers, Dense(hidden_dim => out_dim))
    

    return Chain(layers...) 
end

#scale_output
function scale_antoine_parameters(raw_params)
    # use sigmoid (values between 0 und 1)
    x_sig = sigmoid.(raw_params)
    
    # scale output = sigmoid(x) * (max - min) + min
    A = x_sig[1:1, :] .* (20.0f0 - 5.0f0) .+ 5.0f0
    B = x_sig[2:2, :] .* (6000.0f0 - 1500.0f0) .+ 1500.0f0
    C = x_sig[3:3, :] .* (0.0f0 - (-300.0f0)) .+ (-300.0f0)
    
    return vcat(A, B, C)
end

# ---------------------------------------------------------------------------
# calculate antoine
function calculate_vapor_pressure(scaled_params, temperature)
    A = scaled_params[1:1, :]
    B = scaled_params[2:2, :]
    C = scaled_params[3:3, :]
    
    return A .- B ./ (C .+ temperature .+ 1f-8)
end


# ---------------------------------------------------------------------------
# model
struct GRAPPAModelLux <: LuxCore.AbstractLuxContainerLayer{(:conv0, :conv1, :conv2, :conv3, :pooling, :head)}
    conv0::GATv2Conv
    conv1::GATv2Conv
    conv2::GATv2Conv
    conv3::GATv2Conv
    pooling::MultiHeadAttentionPoolingLux
    head::Chain
end

function GRAPPAModelLux(; node_dim=24, edge_dim=9, conv_dim=32, hidden_dim=16, 
                        heads=2, pooling_heads=1, 
                        num_hidden_layers=3, num_antoine_params=3)
    
    #gnn = build_gnn_gat(node_dim, edge_dim, conv_dim, gnn_heads, num_gnn_layers)
    conv0 = GATv2Conv((node_dim, edge_dim) => conv_dim; heads=heads, concat=false, add_self_loops=false)
    conv1 = GATv2Conv((conv_dim, edge_dim) => conv_dim; heads=heads, concat=false, add_self_loops=false)
    conv2 = GATv2Conv((conv_dim, edge_dim) => conv_dim; heads=heads, concat=false, add_self_loops=false)
    conv3 = GATv2Conv((conv_dim, edge_dim) => conv_dim; heads=heads, concat=false, add_self_loops=false)
    
    pooling = MultiHeadAttentionPoolingLux(conv_dim, conv_dim, pooling_heads)
    
    head = grappa_head(conv_dim + 2, hidden_dim, num_hidden_layers, num_antoine_params)

    return GRAPPAModelLux(conv0, conv1, conv2, conv3, pooling, head)
end


function (model::GRAPPAModelLux)(graph::GNNGraph, ps, st::NamedTuple)
    
    # gnn, input is graph from smiles_to_gnngraph
    #node_out, st_gnn = model.gnn(graph, graph.ndata.x, graph.edata.e, ps.gnn, st.gnn)

    x = graph.ndata.x
    e = graph.edata.e
    
    # gnn, layer separately (since GNNChain cannot process edges)
    x0, st_conv0 = model.conv0(graph, x, e, ps.conv0, st.conv0)

    x0_elu = NNlib.elu.(x0)

    x1, st_conv1 = model.conv1(graph, x0_elu, e, ps.conv1, st.conv1)
    x1_elu = NNlib.elu.(x1)
    
    x2, st_conv2 = model.conv2(graph, x1_elu, e, ps.conv2, st.conv2)
    x2_elu = NNlib.elu.(x2)
    
    x3, st_conv3 = model.conv3(graph, x2_elu, e, ps.conv3, st.conv3)
    node_out = NNlib.elu.(x3)
    
    # pooling
    graph_pooled, st_pool = model.pooling(node_out, ps.pooling, st.pooling)
    
    h_donors = reshape([graph.gdata.h_donors], 1, :)
    h_acceptors = reshape([graph.gdata.h_acceptors], 1, :)
    
    # add h_donors, h_acceptors
    head_input = vcat(graph_pooled, h_donors, h_acceptors)
    
    # prediction head
    antoine_raw, st_head = model.head(head_input, ps.head, st.head)
    
    antoine_scaled = scale_antoine_parameters(antoine_raw)
    
    new_st = (conv0=st_conv0, conv1=st_conv1, conv2=st_conv2, conv3=st_conv3, pooling=st_pool, head=st_head)
    
    return antoine_scaled, new_st
end
