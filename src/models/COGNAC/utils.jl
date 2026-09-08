# Molecule graphs for COGNAClite

const COGNAC_HYBRIDIZATION = Dict(:sp => 1, :sp2 => 2, :sp3 => 3, :sp3d => 4, :sp3d2 => 5)
const COGNAC_BOND_AROMATIC = 4

# node feature vector of an explicit polar hydrogen: Z, degree, shifted charge, hyb, aromatic, nH
const COGNAC_H_FEATURES = (1, 1, 4, 0, 0, 0)

function cognac_molgraph(mol)
    syms = MolecularGraph.atom_symbol(mol)
    heavy = findall(s -> s !== :H, syms)
    node_of = zeros(Int, length(syms))
    node_of[heavy] .= eachindex(heavy)

    z = MolecularGraph.atom_number(mol)
    degrees = heavy_degrees(mol)
    n_hydrogens = MolecularGraph.total_hydrogens(mol)
    hybs = MolecularGraph.hybridization(mol)
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
            get(COGNAC_HYBRIDIZATION, hybs[i], 0),
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

# All molecules in one graph (disconnected, one component per molecule), together with the
# index of the molecule every node belongs to
function cognac_gnngraph(smiles::AbstractVector{<:AbstractString})
    xs, es, src, tgt, molecule_index = Matrix{Int}[], Matrix{Int}[], Int[], Int[], Int[]
    n_nodes = 0
    for (i, s) in enumerate(smiles)
        x, s_i, t_i, e_i = cognac_molgraph(MolecularGraph.smilestomol(s))
        push!(xs, x); push!(es, e_i)
        append!(src, s_i .+ n_nodes); append!(tgt, t_i .+ n_nodes)
        append!(molecule_index, fill(i, size(x, 2)))
        n_nodes += size(x, 2)
    end

    graph = GNNGraph(src, tgt;
        num_nodes = n_nodes,
        ndata = (; x = reduce(hcat, xs)),
        edata = (; e = reduce(hcat, es)),
    )
    return graph, molecule_index
end

# every atom is its own group, unique to this mixture: "<molecule>_<group><element>"
function cognac_group_names(graph::GNNGraph, molecule_index)
    element(z) = MolecularGraph.ATOMTABLE[z]["Symbol"]
    return ["$(molecule_index[k])_$(k)$(element(graph.ndata.x[1, k]))"
            for k in eachindex(molecule_index)]
end

function cognac_groups(components, group_names, molecule_index, grouptype::Symbol)
    return GroupParam(
        [(components[i], [group_names[k] => 1 for k in findall(==(i), molecule_index)])
         for i in eachindex(components)],
        grouptype, String[]
    )
end
