# Molecule graphs and group tables for COGNAC

const COGNAC_HYBRIDIZATION = Dict(:sp => 1, :sp2 => 2, :sp3 => 3, :sp3d => 4, :sp3d2 => 5)
const COGNAC_BOND_AROMATIC = 4

# node feature vector of an explicit polar hydrogen: Z, degree, shifted charge, hyb, aromatic, nH
const COGNAC_H_FEATURES = (1, 1, 4, 0, 0, 0)

# RDKit's hybridization code, which `MolecularGraph.hybridization` assigns only to
# B/C/N/O/Si/P/S, so every other atom falls back to its electron domain count.
function cognac_hybridizations(mol)
    hybs = MolecularGraph.hybridization(mol)
    domains = MolecularGraph.connectivity(mol) .+ MolecularGraph.lone_pair(mol)
    return [get(COGNAC_HYBRIDIZATION, h, clamp(d - 1, 0, 5)) for (h, d) in zip(hybs, domains)]
end

function cognac_molgraph(mol)
    syms = MolecularGraph.atom_symbol(mol)
    heavy = findall(s -> s !== :H, syms)
    node_of = zeros(Int, length(syms))
    node_of[heavy] .= eachindex(heavy)

    z = MolecularGraph.atom_number(mol)
    degrees = heavy_degrees(mol)
    n_hydrogens = MolecularGraph.total_hydrogens(mol)
    hybs = cognac_hybridizations(mol)
    aroms = MolecularGraph.is_aromatic(mol)
    charges = MolecularGraph.atom_charge(mol)

    # atoms whose hydrogens the checkpoint treats as groups of their own
    polar = [i for i in heavy if syms[i] !== :C && n_hydrogens[i] > 0]
    x = Matrix{Int}(undef, 6, length(heavy) + sum(i -> n_hydrogens[i], polar; init=0))

    for (j, i) in enumerate(heavy)
        x[:, j] .= (
            clamp(z[i], 0, 118),
            clamp(degrees[i] + n_hydrogens[i], 0, 7),
            clamp(charges[i], -4, 4) + 4,
            hybs[i],
            aroms[i],
            i in polar ? 0 : clamp(n_hydrogens[i], 0, 5),
        )
    end

    orders = MolecularGraph.bond_order(mol)
    bond_aromatic = MolecularGraph.is_edge_aromatic(mol)
    bond_ring = MolecularGraph.is_edge_in_ring(mol)
    bond_conj = is_conjugated(mol)

    source, target = get_directed_edges(mol)
    src, tgt, e = Int[], Int[], Int[]
    for k in eachindex(source)
        u, v = node_of[source[k]], node_of[target[k]]
        (u > 0 && v > 0) || continue    # bonds to hydrogens written explicitly in the SMILES
        b = (k + 1) ÷ 2                 # `get_directed_edges` emits each bond twice
        push!(src, u); push!(tgt, v)
        push!(e, bond_aromatic[b] ? COGNAC_BOND_AROMATIC : orders[b], bond_conj[b], bond_ring[b])
    end

    # append the polar hydrogens, grouped by parent, the way RDKit's `AddHs` orders them
    j = length(heavy)
    for i in polar, _ in 1:n_hydrogens[i]
        j += 1
        x[:, j] .= COGNAC_H_FEATURES
        push!(src, node_of[i], j); push!(tgt, j, node_of[i])
        push!(e, 1, 0, 0, 1, 0, 0)
    end

    return x, src, tgt, reshape(e, 3, :)
end

# consecutive ids for the distinct keys, numbered by first appearance
function _relabel(keys::AbstractVector)
    ids = Dict{eltype(keys),Int}()
    return [get!(ids, k, length(ids) + 1) for k in keys]
end

# Weisfeiler-Leman classes of one molecule's atoms; atoms of one class get one embedding,
# since a message-passing network is at most as expressive as 1-WL.
function cognac_atom_classes(x, src, tgt, e)
    n = size(x, 2)
    nbrs = [Tuple{Int,Int}[] for _ in 1:n]
    for j in eachindex(src)
        push!(nbrs[tgt[j]], (src[j], e[1, j] + 5 * e[2, j] + 10 * e[3, j]))
    end

    color = _relabel([x[:, k] for k in 1:n])
    for _ in 1:n
        next = _relabel([(color[k], sort!([(color[m], b) for (m, b) in nbrs[k]]))
                         for k in 1:n])
        next == color && break
        color = next
    end

    classes = [Int[] for _ in 1:maximum(color)]
    for k in 1:n
        push!(classes[color[k]], k)
    end
    return classes
end

# all molecules in one disconnected graph, with every node's molecule index and the classes
function cognac_gnngraph(smiles::AbstractVector{<:AbstractString})
    xs, es, src, tgt, molecule_index = Matrix{Int}[], Matrix{Int}[], Int[], Int[], Int[]
    classes = Vector{Int}[]
    n_nodes = 0
    for (i, s) in enumerate(smiles)
        x, s_i, t_i, e_i = cognac_molgraph(MolecularGraph.smilestomol(s))
        push!(xs, x); push!(es, e_i)
        append!(src, s_i .+ n_nodes); append!(tgt, t_i .+ n_nodes)
        append!(molecule_index, fill(i, size(x, 2)))
        # classes are per molecule, so atoms of different molecules are never merged
        append!(classes, (c .+ n_nodes for c in cognac_atom_classes(x, s_i, t_i, e_i)))
        n_nodes += size(x, 2)
    end

    graph = GNNGraph(src, tgt;
        num_nodes = n_nodes,
        ndata = (; x = reduce(hcat, xs)),
        edata = (; e = reduce(hcat, es)),
    )
    return graph, molecule_index, classes
end

# group parameters as the mean over the group's atoms, as in the reference implementation
function cognac_group_params(groups, R, Q, ABC)
    W = zeros(length(groups), length(R))
    for (g, atoms) in enumerate(groups)
        W[g, atoms] .= 1 / length(atoms)
    end
    merged = map(ABC) do M
        X = W * M * W'
        X .= (X .+ X') ./ 2             # symmetric up to roundoff anyway
        X[diagind(X)] .= 0              # aₖₖ = 0, exactly as for a single atom
        X
    end
    return W * R, W * Q, merged
end

# every group is unique to this mixture: "<molecule>_<first atom><element>"
function cognac_group_names(graph::GNNGraph, molecule_index, groups)
    element(z) = MolecularGraph.ATOMTABLE[z]["Symbol"]
    return ["$(molecule_index[k])_$(k)$(element(graph.ndata.x[1, k]))" for k in first.(groups)]
end

# `νₖᵢ` counts the symmetry-equivalent atoms of group `k`, as mod. UNIFAC counts "2 × CH3"
function cognac_groups(components, group_names, molecule_index, groups, grouptype::Symbol)
    group_molecule = [molecule_index[first(g)] for g in groups]
    return GroupParam(
        [(components[i], [group_names[g] => length(groups[g])
                          for g in findall(==(i), group_molecule)])
         for i in eachindex(components)],
        grouptype, String[]
    )
end
