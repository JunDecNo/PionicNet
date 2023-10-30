# 负责统计20中氨基酸中的原子类型
from Bio import PDB

parse = PDB.PDBParser(QUIET=True)
res_dict = {}
structure = parse.get_structure('101mA','E:/OwnCode/PionicNet/data/BioLiP/all_PDB/Receptor/2p0mA.pdb')
if len(structure) > 1:
    print('多链蛋白质')
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 print(f"Model: {model.id}, Chain: {chain.id}, Residue: {residue.get_resname()}, Atom: {atom.get_name()}")
atom = structure[0]['A'][10]
print(f"CA Atom Coordinates: {atom['']}")
# print(atom.get_parent().get_resname())
print(len(structure))
# res = []
    