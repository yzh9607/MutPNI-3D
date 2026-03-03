import pickle
import shutil

from Bio import PDB
from calc_sasa import SASA
from calc_ss import Secondstructure
from calc_depth import calc_depth_noclean
from get_edge_Hbond_angle2 import get_edge
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import os
import torch

import re
from calc_atom_fea import  get_atom_fea

aa2property = {'A':[1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
               'G':[0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
               'V':[3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
               'L':[2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
               'I':[4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
               'F':[2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
               'Y':[2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
               'W':[3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
               'T':[3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
               'S':[1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
               'R':[2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
               'K':[1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
               'H':[2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
               'D':[1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
               'E':[1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
               'N':[1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
               'Q':[1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
               'M':[2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
               'P':[2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
               'C':[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41]}

atom2code_dist = {'C':[1, 0, 0, 0, 0], 'N':[0, 1, 0, 0, 0], 'O':[0, 0, 1, 0, 0], 'S':[0, 0, 0, 1, 0], 'H':[0, 0, 0, 0, 1]}
# atom2code_dist = {'C':[1, 0, 0, 0], 'N':[0, 1, 0, 0], 'O':[0, 0, 1, 0], 'S':[0, 0, 0, 1]}
metal_List = ["ZN", "MG", "FE", "CU", "CA", "NA", "AS", "HG", "MN", "K", "SM", "W", "CO", "NI", "AU", "CD", "PB", "Y",
              "SR", "PT"]

# 定义一个氨基酸的字典，一位代码到三位代码的映射
class AminoAcidDict(dict):
    def __missing__(self, key):
        # If the key is not found, return the key itself
        return key

amino_acid_dict = AminoAcidDict({
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
    'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',
    'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
    'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
})

def aa2code():
    aa2code = {}
    aa_name = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
    for i in range(20):
        code = []
        for j in range(20):
            if i == j:
                code.append(1)
            else:
                code.append(0)
        aa2code[aa_name[i]] = code
    return aa2code

def aa2code_mut():
    aa2code = {}
    aa_name = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', 'X']
    for i in range(21):
        code = []
        for j in range(21):
            if i == j:
                code.append(1)
            else:
                code.append(0)
        aa2code[aa_name[i]] = code
    return aa2code

def get_Hbond(pdb_tag,pdb_path):
    pdb_tag_file = pdb_path +'/'+pdb_tag+'.hb2'
    donor = []
    acceptor = []
    with open(pdb_tag_file, 'r', errors='ignore') as f:
        tag = False
        for line in f.readlines():
            itemlist = line.split()
            if itemlist[-1] == '1' or tag == True:
                tag = True
                d = itemlist[0]
                a = itemlist[2]
                number_d = re.findall(r'\d+', d)
                number_a = re.findall(r'\d+', a)
                try:
                    str_pos_d = str(int(number_d[0]))
                    donor.append(str_pos_d)
                except:
                    print("Hbond_Error:".format(pdb_tag))
                    continue
                try:
                    str_pos_a = str(int(number_a[0]))
                    acceptor.append(str_pos_a)
                except:
                    print("Hbond_Error:".format(pdb_tag))
                    continue
    return donor, acceptor


def feature_alignment1_muchatom_res60(pdb_tag,selected_res, selected_atom, feature_data, res_dict, atom_dict, mut_pos,metal_pos,node_pos_dict,atom_pos_dict,pdb_path):


    res_feature_dict, atom_feature_dict = {}, {}
    donor, acceptor = get_Hbond(pdb_tag,pdb_path)
    try:
        atom_61fea_dict = get_atom_fea(pdb_tag,pdb_path)
    except:
        with open("/home/yanzihao/pytorch/CASTLE/Data/MPD476/allatom_fea_error.txt", "a") as f:
            f.write(f"{pdb_tag}\n")

    sasa_res = feature_data['sasa_res']
    sasa_atom = feature_data['sasa_atom']
    ss_dict = feature_data['ss_dict']
    depth_dict = feature_data['depth_dict']

    aa2code_dict = aa2code()
    aa2code_mut_dict = aa2code_mut()

    ############################### res features ###############################
    for pos in selected_res:
        res_pos = node_pos_dict[pos]
        try:
            res_name = res_dict[pos]
            aa_code = aa2code_dict[res_name]
            if pos == mut_pos:
                res_name_mut = pdb_tag.split('_')[-1]
                aa_code_mut = aa2code_mut_dict[res_name_mut]
            else:
                aa_code_mut = aa_code+[0]
            aa_code_all = aa_code + aa_code_mut
        except:
            aa_code_all = [0]*41
        try:
            ss = ss_dict[pos]
        except:
            ss = [0]*3
        try:
            depth = [depth_dict[pos]]
        except:
            depth = [0]
        try:
            res_name = res_dict[pos]
            properties = aa2property[res_name]
        except:
            properties = [0]*7
        try:
            sasa = [sasa_res[pos]]
        except:
            sasa = [0]

        if pos == mut_pos:
            is_mut = [1]
        else:
            is_mut = [0]

        if pos == metal_pos:
            is_metal = [1]
        else:
            is_metal = [0]

        if pos in donor:
            is_donor = [1]
        else:
            is_donor = [0]

        if  pos in acceptor:
            is_acceptor = [1]
        else:
            is_acceptor = [0]
        res_feature_dict[pos] = res_pos + aa_code_all + ss + depth + properties + sasa + is_donor + is_acceptor + is_mut + is_metal

    for atom_index in selected_atom.keys():
        atom_pos = atom_pos_dict[atom_index]
        try:
            if atom_index in sasa_atom.keys():
                sasa = [sasa_atom[atom_index]]
            else:
                sasa = [sasa_atom[str(selected_atom[atom_index])]]
        except:
            sasa = [0]
        try:
            atom_feature_dict[atom_index] = atom_61fea_dict[int(atom_index)] + atom_pos + sasa
        except:
            atom_feature_dict[atom_index] = [0]*61 +  atom_pos + sasa
            with open("../Data/MPD476/allatom_fea_error.txt", "a") as f:
                f.write(f"{pdb_tag}\n")
            continue

    return res_feature_dict, atom_feature_dict

def get_pdb_array(pdb_file):
    pdb_array = []
    metal_pos_dict = {}
    metal_list = []
    with open(pdb_file, 'r') as pdbfile:
        for line in pdbfile:
            # if line[0:4] == 'ATOM' or line[0:4] == 'HETA':
            if line[0:4] == 'ATOM' or line[0:4] == 'HETA' :
                line_list = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27],
                                line[30:38],
                                line[38:46], line[46:54], line[-4:]]
                line_list = [i.strip() for i in line_list]
                if line_list[0] == 'ATOM' or line_list[0] == 'HETAT':
                    pdb_array.append(line_list)
        pdb_array = np.array(pdb_array, dtype='str')

    return pdb_array


def get_residue_info(pdb_array):
    atom_res_array = pdb_array[:, 6]  # 每一个原子对应的氨基酸编号
    boundary_list = []  # 列表中代表每一个氨基酸的起始原子和终止原子的位置
    pdb_pos_list = []
    start_pointer = 0
    curr_pointer = 0
    curr_atom = atom_res_array[0]

    # One pass through the list of residue numbers and record row number boundaries. Both sides inclusive.
    while (curr_pointer < atom_res_array.shape[0] - 1):
        curr_pointer += 1
        if atom_res_array[curr_pointer] != curr_atom:
            pdb_pos_list.append(curr_atom)
            boundary_list.append([start_pointer, curr_pointer - 1])
            start_pointer = curr_pointer
            curr_atom = atom_res_array[curr_pointer]
    boundary_list.append([start_pointer, atom_res_array.shape[0] - 1])
    pdb_pos_list.append(curr_atom)
    return np.array(boundary_list), pdb_pos_list


def get_nearest_mfs_resindex_noH(pdb_file, mut_pos, wild_aa3):
    selectes_pdb_pos = []
    atompos2index = {}
    new_pdb_array = []
    pdb_array = get_pdb_array(pdb_file)
    residue_index, pdb_pos_list = get_residue_info(pdb_array)
    residue_dm,mut_pos_index = get_edge().get_residue_distance_matrix(pdb_array, residue_index, 'mfs', mut_pos, wild_aa3)
    index = residue_dm.argsort()[mut_pos_index, :]

    for i in index:
        selectes_pdb_pos.append(pdb_pos_list[i])

    index = 0
    metal_pos_inpdb = ''
    for atom_info in pdb_array:
        if atom_info[2][0] == 'C' or atom_info[2][0] == 'N' or atom_info[2][0] == 'O' or atom_info[2][0] == 'S' or atom_info[4] in metal_List :
            if atom_info[6] in selectes_pdb_pos:
                atompos2index[atom_info[1]] = index
                new_pdb_array.append(list(atom_info))
                index += 1
            if atom_info[4] in metal_List :
                metal_pos_inpdb = atom_info[6]
    return selectes_pdb_pos, atompos2index, np.array(new_pdb_array, dtype='str'),metal_pos_inpdb

def get_res_atom_dict(pdb_array):
    res_dict, atom_dict = {}, {}
    for atom_info in pdb_array:
        try:
            aa = three_to_one(atom_info[4])
        except KeyError:
            try:
                aa = atom_info[4]
            except:
                continue
        res_dict[atom_info[6]] = aa
        atom_dict[atom_info[1]] = atom_info[2]
    # print(res_dict, atom_dict)
    return res_dict, atom_dict


def generate_wt_features(pdb_flag, nucleic_acid_pdb_from):
    pdb_file = f'{nucleic_acid_pdb_from}/{pdb_flag}.pdb'

    ############################# structure features #############################
    sasa_res, sasa_atom = SASA(pdb_file)
    ss_dict = Secondstructure(pdb_file)
    depth_dict = ''
    try:
        depth_dict = calc_depth_noclean(pdb_file)
    except:
        with open('../Data/depth_error.txt', 'a') as f:
            f.write('error{}'.format(pdb_flag))
    wt_data = {
        'sasa_res': sasa_res,
        'sasa_atom': sasa_atom,
        'ss_dict': ss_dict,
        'depth_dict': depth_dict,
    }

    return wt_data


def generate_node_feature(feature_dict, index_pos_dict, feature_num):
    residue_node_feature = np.empty((len(index_pos_dict), feature_num), dtype=np.float64)
    all_pdbpos = index_pos_dict.values()
    for i, pdbpos in enumerate(all_pdbpos):
        residue_node_feature[i] = [float(feature) for feature in feature_dict[pdbpos]]
    return residue_node_feature

def generate_input(pdb2mutation, outfile):
    all_pdb = []
    all_label = []
    with open(outfile, 'w') as f_r:
        for pdb_flag in pdb2mutation:

            pdb_file_wt = f'{nucleic_acid_pdb_from}/{pdb_flag}.pdb'
            nucleic_acid_type = pdb_flag.split('_')[0]
            chain_id = pdb_flag.split('_')[5]
            mut_pos = pdb_flag.split('_')[3]
            pdb_id = pdb_flag.split('_')[1]
            pdb_i = pdb_flag.split('_')[2]
            wild_aa = pdb_flag.split('_')[4]
            mut_aa = pdb_flag.split('_')[6]
            wild_aa3 = amino_acid_dict[wild_aa]

            try:
                data_wt = generate_wt_features(pdb_flag, nucleic_acid_pdb_from)  # 改地址
            except OSError:
                continue

            ############################# generating wild input #####################################
            try:
                _, _, pdb_array_wt, metal_pos = get_nearest_mfs_resindex_noH(pdb_file_wt, mut_pos, wild_aa3)
            except:
                print("错误样本为为为为：{}".format(pdb_flag))
                continue

            try:
                # 改地址
                pdb_array_wt, res_edge_index_wt, res_edge_feature_wt, atom_res_index_wt, node_pos, res_index_pos_dict_wt, \
                    atom_index_pos_wt, node_pos_dict, atom_pos_dict, basic_attn, basic_attn_peiwei \
                    = get_edge().generate_mfs_residue_edge_feature(pdb_array_wt, pdb_flag, nucleic_acid_pdb_from, mut_pos, wild_aa3)
            except:
                continue
            res_dict_wt, atom_dict_wt = get_res_atom_dict(pdb_array_wt)
            selected_res_wt = {value:key for key,value in res_index_pos_dict_wt.items()}
            selected_atom_wt = {value:key for key,value in atom_index_pos_wt.items()}
            try:
                res_feat_dict_wt, atom_feat_dict_wt = feature_alignment1_muchatom_res60(pdb_flag, selected_res_wt,
                                                                                  selected_atom_wt,
                                                                                  data_wt, res_dict_wt, atom_dict_wt,
                                                                                  mut_pos, metal_pos, node_pos_dict,
                                                                                  atom_pos_dict,nucleic_acid_pdb_from)
            except:
                continue

            atom_edge_index_wt, atom_edge_feature_wt = get_edge().generate_atom_edge_feature(pdb_array_wt)
            res_node_feat_wt = generate_node_feature(res_feat_dict_wt, res_index_pos_dict_wt, 60)
            atom_node_feat_wt = generate_node_feature(atom_feat_dict_wt, atom_index_pos_wt, 65)

            res_node_wt_path = f'{res_node_from_filder}/{pdb_flag}.pt'
            res_edge_wt_path = f'{res_edge_from_filder}/{pdb_flag}.pt'
            res_edge_index_wt_path = f'{res_index_from_filder}/{pdb_flag}.pt'
            atom_node_wt_path = f'{atom_node_from_filder}/{pdb_flag}.pt'
            atom_edge_wt_path = f'{atom_edge_from_filder}/{pdb_flag}.pt'
            atoms_edge_index_wt_path = f'{atom_index_from_filder}/{pdb_flag}.pt'
            atom2res_index_wt_path = f'{atom2res_from_filder}/{pdb_flag}.pt'
            node_pos_wt_path = f'{node_pos_from_filder}/{pdb_flag}.pt'
            basic_attn_peiwei_wt_path = f'{basic_attn_peiwei_from_filder}/{pdb_flag}.pt'

            res_node_feat_wt = torch.tensor(res_node_feat_wt,dtype=torch.float)
            res_edge_feature_wt = torch.tensor(res_edge_feature_wt,dtype=torch.float)
            res_edge_index_wt = torch.tensor(res_edge_index_wt,dtype=torch.int32)
            atom_node_feat_wt = torch.tensor(atom_node_feat_wt,dtype=torch.float)
            atom_edge_feature_wt = torch.tensor(atom_edge_feature_wt,dtype=torch.float)
            atom_edge_index_wt = torch.tensor(atom_edge_index_wt,dtype=torch.int64)
            atom_res_index_wt = torch.tensor(atom_res_index_wt,dtype=torch.int32)
            node_pos_wt = torch.tensor(node_pos, dtype=torch.float)
            basic_attn_peiwei_wt = torch.tensor(basic_attn_peiwei, dtype=torch.float)

            torch.save(res_node_feat_wt,res_node_wt_path)
            torch.save(res_edge_feature_wt,res_edge_wt_path)
            torch.save(res_edge_index_wt,res_edge_index_wt_path)
            torch.save(atom_node_feat_wt,atom_node_wt_path)
            torch.save(atom_edge_feature_wt,atom_edge_wt_path)
            torch.save(atom_edge_index_wt,atoms_edge_index_wt_path)
            torch.save(atom_res_index_wt,atom2res_index_wt_path)
            torch.save(node_pos_wt, node_pos_wt_path)
            torch.save(basic_attn_peiwei_wt, basic_attn_peiwei_wt_path)

            all_pdb.append('{}_{}_{}_{}_{}_{}_{}'.format(nucleic_acid_type,pdb_id.lower(),pdb_i,mut_pos,wild_aa,chain_id,mut_aa))
            f_r.write(nucleic_acid_type + '\t' + pdb_id + '\t' + pdb_i + '\t' + chain_id + '\t' + mut_pos + '\t' + wild_aa + '/' + mut_aa +  '\n')

    print("总共生成{}条数据".format(len(all_pdb)))
    for x in pdb2mutation:
        if x not in all_pdb:
            with open('../Data/MPD476/nogetfea_samples.txt', 'a') as f:
                f.write(x)
                f.write('\n')

def get_dataset(dataset_path):
    pdb2mutations = []
    with open(dataset_path, 'r') as f_r:
        for pdb_flag in f_r:
            pdb2mutations.append(pdb_flag[:-1])
    return pdb2mutations

def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！

    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


save_dir = '../Data/MPD476/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

res_node_from_filder = save_dir + "res_node_fea"
res_edge_from_filder = save_dir+'res_edge_fea'
res_index_from_filder = save_dir+"res_index_fea"
atom_node_from_filder = save_dir+"atom_node_fea"
atom_edge_from_filder = save_dir+"atom_edge_fea"
atom_index_from_filder = save_dir+"atom_index_fea"
atom2res_from_filder = save_dir+"atom2res_fea"
node_pos_from_filder = save_dir+"node_pos_from"
basic_attn_peiwei_from_filder = save_dir+"basic_attn_peiwei"


nucleic_acid_pdb_from = '../Data/MPD476/'+'nucleic_acid_pdb'
label_filder = '../Data/MPD476/'+"label"

RemoveDir(basic_attn_peiwei_from_filder)
RemoveDir(res_node_from_filder)
RemoveDir(res_edge_from_filder)
RemoveDir(atom_node_from_filder)
RemoveDir(atom2res_from_filder)
RemoveDir(node_pos_from_filder)
RemoveDir(atom_edge_from_filder)
RemoveDir(atom_index_from_filder)
RemoveDir(res_index_from_filder)

data_path = '../Data/MPD476/record_data.txt'
dataout_path = '../Data/MPD476/get_record_data.txt'

# 打开包含pdb文件名的txt文件
with open(data_path, 'r') as file:
    # 逐行读取文件名
    for line in file:
        # 去除行尾的换行符
        pdb_file = line.strip()
        if pdb_file == '' or os.path.exists(f'{nucleic_acid_pdb_from}/{pdb_file}.hb2'):
            continue
        # 构造HBPLUS命令 按照官网进行HBPLUS的安装和路径修改
        file_path = os.path.abspath(f'{nucleic_acid_pdb_from}/{pdb_file}.pdb')
        command = f'/home/hbplus/hbplus {file_path}'
        # 在命令行中执行命令
        os.system(command)
        # 移动文件
        shutil.move(f'{pdb_file}.hb2', nucleic_acid_pdb_from)

pdb2mutations = get_dataset(data_path)
pdb2mutations = [x.strip() for x in pdb2mutations if x.strip()!='']
generate_input(pdb2mutations, dataout_path)



