const HOMO_ATOMS = [:C, :H]
const HALOGEN_ATOMS = [:F, :Cl, :Br, :I]

const RDKIT_HBA = MolecularGraph.smartstomol(raw"[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]")
const RDKIT_HBD = MolecularGraph.smartstomol(raw"[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]")

function get_descriptors(smiles::AbstractString)
    clean_smiles = smiles
    
    clean_smiles = replace(clean_smiles, "N(=O)=O" => "[N+](=O)[O-]")
    clean_smiles = replace(clean_smiles, "O=N(=O)" => "O=[N+]([O-])")
    clean_smiles = replace(clean_smiles, "n(=O)" => "[n+]([O-])")
    clean_smiles = replace(clean_smiles, "C=N#N" => "[C-]=[N+]=[N-]")

    mol = MolecularGraph.smilestomol(clean_smiles)
    
    # check if smiles is a valid molecule
    if isnothing(mol)
        return nothing
    end

    # get atoms
    atoms = atom_symbol(mol)

    hba_matches = substruct_matches(mol, RDKIT_HBA)
    hbd_matches = substruct_matches(mol, RDKIT_HBD)

    return Dict(
        "exactmw" => Float64(exact_mass(mol)),
        "NumRings" => Int64.(length(sssr(mol))),
        "NumHeteroatoms" => Int64(count(a -> a ∉ HOMO_ATOMS, atoms)),
        "NumHeavyAtoms" => Int64(heavy_atom_count(mol)),
        "NumHBA" => Int64(length(collect(hba_matches))),
        "NumHBD" => Int64(length(collect(hbd_matches))),
        "NumHalogens" => Int64(count(a -> a ∈ HALOGEN_ATOMS, atoms))
    )
end
