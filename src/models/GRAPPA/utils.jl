using GNNGraphs: GNNGraph
using Graphs
using Lux
using LuxCore
using MolecularGraph
using Random
using Statistics

include("../../utils/bond_features.jl")
using .BondFeatures

#include("../../layers/utils.jl")

const GRAPPA_ATOMS = [:C, :N, :O, :Cl, :S, :F, :Br, :I, :P]
const GRAPPA_HYB = [:S, :SP, :SP2, :SP3]
const GRAPPA_BONDS = [0, 1, 2, 3, 4]
const GRAPPA_BOND_TYPES = [1, 2, 3]
const GRAPPA_HS = [0, 1, 2, 3]


function atom_feature(mol)
    num_atoms = nv(mol)
    # empty matrix with 24 Properties x Number of nodes
    features = zeros(Float32, 24, num_atoms)

    # throw error if atom has formal charge
    if any(atom_charge(mol) .!= 0)
        error("Atom has formal charge!")
    end
    # throw error if atom has radical electrons
    # if atom_radical_electrons(mol) != 0   this function doesnt exist (yet)
    #     error("Atom has radical electrons!")
    # end

    # get the whole chemical info
    syms = atom_symbol(mol)
    rings = is_in_ring(mol)
    aroms = is_aromatic(mol)
    hybs = hybridization(mol)
    #degs = Graphs.degree(mol)
    hs = total_hydrogens(mol) 
    
    for i in 1:num_atoms
        # atom-typ
        f_type = onehot_encoder(syms[i], GRAPPA_ATOMS)
        # ring
        f_ring = Float32[rings[i] ? 1.0 : 0.0]
        # aromcatic
        f_arom = Float32[aroms[i] ? 1.0 : 0.0]
        # hybridisation
        current_hyb = Symbol(uppercase(string(hybs[i])))

        # hybridisation fix for halogens (RDKit calculates SP3 for halogens, MolecularGraph nothing)
        if syms[i] ∈ [:F, :Cl, :Br, :I] && current_hyb ∉ GRAPPA_HYB
            current_hyb = :SP3
        end
        f_hyb  = onehot_encoder(current_hyb, GRAPPA_HYB)
        
        # amount connections
        heavy_degree = 0    # count only heavy atom neighbors like RDKit
        for neighbor_idx in neighbors(mol, i)
            if syms[neighbor_idx] != :H
                heavy_degree += 1
            end
        end
        f_bond = onehot_encoder(heavy_degree, GRAPPA_BONDS)
        # amount H-atoms
        f_h    = onehot_encoder(hs[i], GRAPPA_HS)
        
        # add all the information
        atom_vec = vcat(f_type, f_ring, f_arom, f_hyb, f_bond, f_h)
        features[:, i] = atom_vec
    end
    
    return features 
end


function bond_feature(mol)
    raw_features = BondFeatures.get_all_bond_features(mol)
    features = copy(raw_features')
    return features
end


function molgraph_to_gnngraph(mol; target=nothing)
    syms_full = MolecularGraph.atom_symbol(mol)
    
    # identificate heavy atoms (== without H)
    heavy_indices = findall(s -> s != :H, syms_full)
    num_nodes = length(heavy_indices)
    
    # only load features for this atoms
    atom_features = atom_feature(mol)[:, heavy_indices]
    
    # load edges and filter them
    src_orig, tgt_orig = get_directed_edges(mol)
    bond_features_orig = bond_feature(mol)
    
    # mapping from old to new indices after filtering out H-atoms
    old_to_new = zeros(Int, length(syms_full))
    for (new_idx, old_idx) in enumerate(heavy_indices)
        old_to_new[old_idx] = new_idx
    end
    
    source = Int[]
    target_nodes = Int[]
    edge_keep = Int[]
    
    for i in eachindex(src_orig)
        u_new = old_to_new[src_orig[i]]
        v_new = old_to_new[tgt_orig[i]]
        
        # only use edges, which are between heavy atoms
        if u_new > 0 && v_new > 0
            push!(source, u_new)
            push!(target_nodes, v_new)
            push!(edge_keep, i)
        end
    end
    
    bond_features = bond_features_orig[:, edge_keep]

    # self loops
    num_features = size(bond_features, 1)
    self_loop_features = zeros(Float32, num_features, num_nodes)
    incoming_counts = zeros(Int, num_nodes)

    for e in eachindex(target_nodes)
        v = target_nodes[e]
        self_loop_features[:, v] .+= bond_features[:, e]
        incoming_counts[v] += 1
    end

    for v in 1:num_nodes
        if incoming_counts[v] > 0
            self_loop_features[:, v] ./= incoming_counts[v]
        end
    end

    source = vcat(source, collect(1:num_nodes))
    target_nodes = vcat(target_nodes, collect(1:num_nodes))
    bond_features = hcat(bond_features, self_loop_features)

    # adjacency matrix (num_nodes x num_nodes)
    adjacency = zeros(Float32, num_nodes, num_nodes)
    for i in eachindex(source)
        u, v = source[i], target_nodes[i]
        adjacency[u, v] = 1.0f0
    end
    
    # global features
    aroms_full = MolecularGraph.is_aromatic(mol)
    acc_count = 0.0
    for i in eachindex(syms_full)
        # use syms_full, because RDKit (Grappa Python reference) counts acceptors before deleting H-atoms
        if syms_full[i] ∈ [:N, :O]
            acc_count += 1.0
        elseif syms_full[i] == :S && !aroms_full[i]
            acc_count += 1.0
        end
    end
    
    h_acceptors = Float32(acc_count)
    h_donors = Float32(MolecularGraph.hydrogen_donor_count(mol))

    # create GNNGraph
    graph = GNNGraph(source, target_nodes, 
                    ndata=(; x=atom_features), 
                    edata=(; e=bond_features), 
                    gdata=(; adj=adjacency, h_donors=h_donors, h_acceptors=h_acceptors))

    if target !== nothing
        graph = GNNGraph(graph, gdata=(; adj=adjacency, h_donors=h_donors, h_acceptors=h_acceptors, y=Float32[target]))
    end

    return graph
end


function smiles_to_gnngraph(smiles::AbstractString; target=nothing)
    mol = smiles_to_molgraph(smiles)
    return molgraph_to_gnngraph(mol; target)
end
