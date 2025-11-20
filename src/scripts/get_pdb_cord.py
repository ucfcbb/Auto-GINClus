from Bio.PDB import MMCIFParser
from functools import reduce


def centroid(coord_list):
    if len(coord_list) > 0:
        return list(map(lambda z: 1.*z/len(coord_list), reduce(lambda x, y: (x[0]+y[0], x[1]+y[1], x[2]+y[2]), coord_list)))
    return None


def get_backbone_and_sugar_atoms():
    
    backbone_atoms = ["C3'", "C4'", "C5'", "O3'", "O5'", "P"]
    sugar_atoms = ["C1'", "C2'", "C3'", "C4'", "O4'"]
    # base_atoms = ["C1'", "C2", "C4", "C5", "C6", "C8", "N1", "N3", "N7", "N9"]
    phosphate_bead = ["P", "OP1", "OP2"]

    return backbone_atoms, sugar_atoms


def get_atom_coordinate(pdb_id, cur_chain, pdb_fn, residue_list):

    parser = MMCIFParser()
    structure = parser.get_structure("1s72", "input_files/1s72.cif")
    backbone_atoms, sugar_atoms = get_backbone_and_sugar_atoms()

    backbone = {}
    sugar = {}
    

    for model in structure:
        
        for chain in model:
            # Example: Targeting a specific residue (e.g., residue 10, histidine in chain A)
            if chain.id == cur_chain:  # Check for chain A

                my_residues = []

                for residue in chain:
                    my_residues.append(residue.id[1])

                for r in residue_list:
                    if r not in my_residues:
##                        backbone[r] = [0]
                        print(f"Residue: {r} doesn't exist, Coordinates: 0")
                        continue
                
                for residue in chain:
                    if residue.id[1] in residue_list:
                        print("residue: ", residue.id[1])
                        backbone_atom_coord = []
                        for atom in residue:
                            if atom.name in backbone_atoms:
                                # Access atom properties like name and coordinates
                                print(f"Residue: {residue.resname}, Atom: {atom.name}, Coordinates: {atom.get_coord()}") # Use .get_coord() for the coordinates
                                backbone_atom_coord.append(atom.get_vector())
                        backbone[residue.id[1]] = centroid(backbone_atom_coord)

    print(backbone)

                        

pdb_id = "1s72"
chain = "0"
pdb_fn = "input_files/1s72.cif"
residue_list = [10, 11, 12, 15, 16, 17]

get_atom_coordinate(pdb_id, chain, pdb_fn, residue_list)
