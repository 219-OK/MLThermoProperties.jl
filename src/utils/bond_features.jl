# Bond stereo categories (matching RDKit's possible_stereo list)
const STEREO_NONE = 0
const STEREO_Z    = 1  # cis
const STEREO_E    = 2  # trans

const FEATURE_LABELS = [
    "is_single", "is_double", "is_triple", "is_aromatic",
    "is_conjugated", "is_in_ring",
    "stereo_NONE", "stereo_Z", "stereo_E"
]

# Atoms whose electrons are available for conjugation
const CONJUGATION_ATOMS = (:B, :C, :N, :O, :Si)

# Default (lowest) valence per element, used to detect hypervalent atoms.
const DEFAULT_VALENCE = Dict(
    :B => 3, :C => 4, :N => 3, :O => 2, :Si => 4, :P => 3, :S => 2,
    :F => 1, :Cl => 1, :Br => 1, :I => 1,
)

function is_conjugated(mol)
    syms = atom_symbol(mol)
    val = valence(mol)
    conn = connectivity(mol)
    lp = lone_pair(mol)
    hs = total_hydrogens(mol)
    orders = bond_order(mol)
    arom = is_edge_aromatic(mol)
    ernk = MolecularGraph.edge_rank(mol)

    # atoms that can contribute electrons to a conjugated system
    cand = [syms[i] ∈ CONJUGATION_ATOMS && (lp[i] > 0 || val[i] > conn[i]) for i in vertices(mol)]

    result = falses(ne(mol))
    for i in vertices(mol)
        cand[i] || continue
        nbrs = neighbors(mol, i)
        # RDKit only conjugates atoms with two or three substituents
        2 <= count(w -> syms[w] != :H, nbrs) + hs[i] <= 3 || continue
        for u in nbrs
            e1 = MolecularGraph.edge_rank(ernk, i, u)
            (orders[e1] > 1 || arom[e1]) || continue
            # RDKit does not conjugate through a hypervalent partner (P ylides, sulfoximines)
            val[u] <= get(DEFAULT_VALENCE, syms[u], 0) || continue
            for w in nbrs
                (w == u || !cand[w]) && continue
                result[e1] = true
                result[MolecularGraph.edge_rank(ernk, i, w)] = true
            end
        end
    end
    return result
end

# Hybridization following RDKit: a heteroatom is only promoted from sp3 to sp2 if it
# carries a conjugated bond (MolecularGraph promotes next to any sp/sp2 neighbor), and
# halogens are SP3 (MolecularGraph assigns them nothing).
_spn(n) = n == 4 ? :SP3 : n == 3 ? :SP2 : n == 2 ? :SP : :none

function rdkit_hybridization(mol)
    syms = atom_symbol(mol)
    val = valence(mol)
    conn = connectivity(mol)
    lp = lone_pair(mol)
    conj = is_conjugated(mol)
    ernk = MolecularGraph.edge_rank(mol)

    hybs = fill(:none, nv(mol))
    for i in vertices(mol)
        if syms[i] ∈ (:F, :Cl, :Br, :I)
            hybs[i] = :SP3
        elseif syms[i] ∈ (:B, :C, :N, :O, :Si, :P, :S)
            hybs[i] = _spn(conn[i] + lp[i])
        end
    end
    for i in vertices(mol)
        syms[i] ∈ (:O, :N, :S) || continue
        (hybs[i] === :SP3 && val[i] < 4 && lp[i] > 0) || continue
        any(w -> conj[MolecularGraph.edge_rank(ernk, i, w)], neighbors(mol, i)) && (hybs[i] = :SP2)
    end
    return hybs
end

# absolute E/Z assignment 
function _cip_expand(mol, orders, ernk, hs, frontier)
    next = Tuple{Int,Int,Bool}[]
    for (v, p, isdup) in frontier
        (isdup || v == 0) && continue
        # a multiple bond duplicates the atom at both of its ends
        append!(next, ((p, v, true) for _ in 2:orders[MolecularGraph.edge_rank(ernk, v, p)]))
        for w in neighbors(mol, v)
            w == p && continue
            push!(next, (w, v, false))
            e = MolecularGraph.edge_rank(ernk, v, w)
            append!(next, ((w, v, true) for _ in 2:orders[e]))
        end
        append!(next, ((0, v, true) for _ in 1:hs[v]))
    end
    return next
end

# Compare substituents `a` and `b` of `root` sphere by sphere (1: a wins, -1: b wins, 0: tie)
function cip_rank(mol, a, b, root; maxdepth=12, maxwidth=4096)
    anum = atom_number(mol)
    orders = bond_order(mol)
    hs = implicit_hydrogens(mol)
    ernk = MolecularGraph.edge_rank(mol)
    # atomic numbers of one sphere, highest first; v == 0 marks an implicit hydrogen
    numbers(f) = sort!([v == 0 ? 1 : anum[v] for (v, _, _) in f]; rev=true)

    fa = [(a, root, false)]
    fb = [(b, root, false)]
    for _ in 1:maxdepth
        (isempty(fa) && isempty(fb)) && return 0
        c = cmp(numbers(fa), numbers(fb))   # lexicographic: the shorter sphere loses
        c == 0 || return c
        (length(fa) > maxwidth || length(fb) > maxwidth) && return 0
        fa = _cip_expand(mol, orders, ernk, hs, fa)
        fb = _cip_expand(mol, orders, ernk, hs, fb)
    end
    return 0
end

# Highest ranked neighbor of `atom`, ignoring `exclude`; `nothing` if the two rank equally
function cip_winner(mol, atom, exclude)
    nbrs = [w for w in neighbors(mol, atom) if w != exclude]
    # the second substituent is an implicit hydrogen and always ranks lowest
    length(nbrs) == 1 && return atom_symbol(mol)[only(nbrs)] === :H ? nothing : only(nbrs)
    length(nbrs) == 2 || return nothing
    c = cip_rank(mol, nbrs[1], nbrs[2], atom)
    return c == 0 ? nothing : nbrs[c > 0 ? 1 : 2]
end

function bond_stereo(mol, e, sb)
    hi_src = cip_winner(mol, src(e), dst(e))
    hi_dst = cip_winner(mol, dst(e), src(e))
    (isnothing(hi_src) || isnothing(hi_dst)) && return STEREO_NONE
    is_cis = sb.is_cis ⊻ (hi_src != sb.first) ⊻ (hi_dst != sb.second)
    return is_cis ? STEREO_Z : STEREO_E
end

# Extract bond features for every bond in both directions (i→j and j→i)
function get_all_bond_features(mol)
    orders  = bond_order(mol)
    arom    = is_edge_aromatic(mol)
    in_ring = is_edge_in_ring(mol)
    conj    = is_conjugated(mol)

    ernk = MolecularGraph.edge_rank(mol)
    stereo = fill(STEREO_NONE, ne(mol))
    for (edge, sb) in mol[:stereobond]
        stereo[ernk[edge]] = bond_stereo(mol, edge, sb)
    end

    result = Matrix{Float32}(undef, 2 * ne(mol), length(FEATURE_LABELS))
    for i in 1:ne(mol)
        a, s = arom[i], stereo[i]
        fv = Float32[
            !a && orders[i] == 1, !a && orders[i] == 2, !a && orders[i] == 3, a,
            conj[i], in_ring[i],
            s == STEREO_NONE, s == STEREO_Z, s == STEREO_E,
        ]
        result[2i - 1, :] .= fv
        result[2i,     :] .= fv
    end
    return result
end
