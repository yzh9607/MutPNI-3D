from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


# 读取 PDB 文件并创建 RDKit 分子对象
def read_pdb_file(file_path):
    return Chem.MolFromPDBFile(file_path)


# 将类别特征进行one-hot编码
def one_hot_encode(value, categories):
    encoding = np.zeros(len(categories))
    if value in categories:
        encoding[categories.index(value)] = 1
    return encoding.tolist()


# 获取原子特征
def get_atom_features(mol):
    atom_features = {}
    for atom in mol.GetAtoms():
        pos = atom.GetIdx()
        atom_serial = atom.GetPDBResidueInfo().GetSerialNumber()
        features = []

        features.extend(one_hot_encode(atom.GetSymbol(), ['C', 'N', 'O', 'S','other']))
        features.extend(one_hot_encode(atom.GetDegree(), [0, 1, 2, 3, 4, 5]))
        features.extend(one_hot_encode(atom.GetFormalCharge(), [-2, -1, 1, 2, 0]))
        features.extend(one_hot_encode(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                                                             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                                             Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                                             'other']))
        features.extend(one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
        features.extend(one_hot_encode(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                                 Chem.rdchem.HybridizationType.SP2,
                                                                 Chem.rdchem.HybridizationType.SP3,
                                                                 Chem.rdchem.HybridizationType.SP3D,
                                                                 Chem.rdchem.HybridizationType.SP3D2]))
        features.extend(one_hot_encode(get_period(atom.GetAtomicNum()), [1, 2, 3, 4, 5, 6]))
        features.extend(one_hot_encode(get_group(atom.GetAtomicNum()), list(range(1, 19))))

        features.append(atom.GetMass() / 100)
        features.append(atom.GetExplicitValence() / 10)
        features.append(atom.GetImplicitValence() / 10)
        features.append(get_vdw_radius(atom.GetSymbol()))
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(1 if is_acceptor(atom) else 0)
        features.append(1 if is_donor(atom) else 0)

        atom_features[atom_serial] = features

    return atom_features


# 获取元素在周期表中的周期（行号）
def get_period(atomic_number):
    if atomic_number <= 2:
        return 1
    elif atomic_number <= 10:
        return 2
    elif atomic_number <= 18:
        return 3
    elif atomic_number <= 36:
        return 4
    elif atomic_number <= 54:
        return 5
    elif atomic_number <= 86:
        return 6
    else:
        return 7


# 获取元素在周期表中的族（列号）
def get_group(atomic_number):
    if atomic_number == 1:
        return 1
    elif atomic_number == 2:
        return 18
    elif atomic_number <= 10:
        return atomic_number - 2 + 1
    elif atomic_number <= 18:
        return atomic_number - 10
    elif atomic_number <= 36:
        return atomic_number - 18
    elif atomic_number <= 54:
        return atomic_number - 36
    elif atomic_number <= 86:
        return atomic_number - 54
    else:
        return 18


# 获取范德华半径
def get_vdw_radius(symbol):
    vdw_radii = {
        'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'P': 1.8,
        'S': 1.8, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'He': 1.4, 'Ne': 1.54,
        'Ar': 1.88, 'Kr': 2.02, 'Xe': 2.16
    }
    return vdw_radii.get(symbol, 2.0)


# 判断是否为氢键受体
def is_acceptor(atom):
    acceptor_types = ['O', 'N']
    return atom.GetSymbol() in acceptor_types and atom.GetTotalNumHs() < 2


# 判断是否为氢键供体
def is_donor(atom):
    donor_types = ['N', 'O']
    return atom.GetSymbol() in donor_types and atom.GetTotalNumHs() > 0



def get_atom_fea(pdb_tag,pdb_path):
    pdb_file_path = pdb_path + '/' + pdb_tag + '.pdb'
    mol = read_pdb_file(pdb_file_path)
    features = get_atom_features(mol)
    return features
