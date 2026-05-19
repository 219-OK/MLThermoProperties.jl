# Bond stereo categories (matching RDKit's possible_stereo list)
const STEREO_NONE = 0
const STEREO_Z    = 1  # cis
const STEREO_E    = 2  # trans
const POSSIBLE_STEREO = [STEREO_NONE, STEREO_Z, STEREO_E]

const FEATURE_LABELS = [
    "is_single", "is_double", "is_triple", "is_aromatic",
    "is_conjugated", "is_in_ring",
    "stereo_NONE", "stereo_Z", "stereo_E"
]

bond_feature_labels() = FEATURE_LABELS

# One-hot encode val among possible_values
function one_of_k_encoding(val, possible_values)
    encoding = zeros(Float32, length(possible_values))
    idx = findfirst(==(val), possible_values)
    isnothing(idx) && error("input $val not in allowable set $possible_values")
    encoding[idx] = 1.0f0
    return encoding
end

function is_conjugated(mol)
    hyb = hybridization(mol)
    edge_list = collect(edges(mol))
    n = ne(mol)

    # Find bonds where both endpoints are sp2 or sp
    sp2_bond = falses(n)
    for (i, e) in enumerate(edge_list)
        sp2_bond[i] = (hyb[src(e)] in (:sp2, :sp)) && (hyb[dst(e)] in (:sp2, :sp))
    end

    # Build adjacency among sp2/sp bonds (two bonds are adjacent if they share a vertex)
    parent = collect(1:n)
    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra != rb && (parent[ra] = rb)
    end

    # For each vertex, collect incident sp2/sp bond indices and union them
    vertex_bonds = [Int[] for _ in 1:Graphs.nv(mol)]
    for (i, e) in enumerate(edge_list)
        sp2_bond[i] || continue
        push!(vertex_bonds[src(e)], i)
        push!(vertex_bonds[dst(e)], i)
    end
    for vb in vertex_bonds
        for j in 2:length(vb)
            union!(vb[1], vb[j])
        end
    end

    # Count component sizes
    comp_size = Dict{Int,Int}()
    for i in 1:n
        sp2_bond[i] || continue
        r = find(i)
        comp_size[r] = get(comp_size, r, 0) + 1
    end

    # A bond is conjugated if its component has ≥ 2 bonds
    result = falses(n)
    for i in 1:n
        sp2_bond[i] || continue
        result[i] = comp_size[find(i)] >= 2
    end
    return result
end

# Extract bond features for every bond in both directions (i→j and j→i)
function get_all_bond_features(mol)
    orders   = bond_order(mol)
    arom     = is_edge_aromatic(mol)
    in_ring  = is_edge_in_ring(mol)
    conj     = is_conjugated(mol)

    ernk = MolecularGraph.edge_rank(mol)
    stereo_types = fill(STEREO_NONE, ne(mol))
    for (edge, sb) in mol[:stereobond]
        stereo_types[ernk[edge]] = sb.is_cis ? STEREO_Z : STEREO_E
    end

    n_feats = length(FEATURE_LABELS)
    result = Matrix{Float32}(undef, 2 * ne(mol), n_feats)

    for (i, e) in enumerate(edges(mol))
        bo       = orders[i]
        is_arom  = arom[i]

        is_single = !is_arom && bo == 1
        is_double = !is_arom && bo == 2
        is_triple = !is_arom && bo == 3

        stereo_enc = one_of_k_encoding(stereo_types[i], POSSIBLE_STEREO)

        fv = Float32[
            is_single, is_double, is_triple, is_arom,
            conj[i], in_ring[i], stereo_enc...
        ]
        result[2i - 1, :] .= fv
        result[2i,     :] .= fv
    end
    return result
end

get_all_bond_features(smiles::AbstractString) = get_all_bond_features(smilestomol(smiles))
