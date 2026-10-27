## Molecule graph helpers shared by the GNN-based models

# Every bond in both directions, in `edges(mol)` order (bond b occupies positions 2b-1, 2b)
function get_directed_edges(mol)
    source = Int[]
    target_nodes = Int[]
    for edge in Graphs.edges(mol)
        u = Graphs.src(edge)
        v = Graphs.dst(edge)
        push!(source, u); push!(target_nodes, v) # forward
        push!(source, v); push!(target_nodes, u) # backward
    end

    return source, target_nodes
end

# Number of non-hydrogen neighbors per atom (RDKit's heavy-atom degree)
function heavy_degrees(mol)
    syms = MolecularGraph.atom_symbol(mol)
    return [count(j -> syms[j] !== :H, Graphs.neighbors(mol, i)) for i in 1:Graphs.nv(mol)]
end
