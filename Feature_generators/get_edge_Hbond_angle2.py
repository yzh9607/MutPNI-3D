import re

import math

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import itertools
import os

metal_List = ["ZN", "MG", "FE", "CU", "CA", "NA", "AS", "HG", "MN", "K", "SM", "W", "CO", "NI", "AU", "CD", "PB", "Y",
              "SR", "PT"]
def get_Hbond(pdb_tag,pdbdir_path):
    pdb_tag_file = pdbdir_path +'/'+pdb_tag+'.hb2'
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


zheng_res = ['LYS','ARG','HIS']
fu_res = ['ASN','GIN','SER','THR','TYR','CYS','ASP','GLU']

# 疏水性氨基酸列表
HYDROPHOBIC_res = ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'VAL', 'PRO', 'TRP']
# 非疏水性氨基酸列表
NON_HYDROPHOBIC_res = ['LYS', 'ARG', 'HIS', 'ASN', 'GLN', 'SER', 'THR', 'TYR', 'CYS', 'ASP', 'GLU']

class get_edge(object):
    def __init__(self):
        super(get_edge, self).__init__()

    def get_residue_info(self, pdb_array):
        atom_res_array = pdb_array[:, 6]  # 每一个原子对应的氨基酸编号
        # print(atom_res_array)
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
        pdb_pos_dict = {i: pdb_pos_list[i] for i in range(0,len(pdb_pos_list))}
        return np.array(boundary_list), pdb_pos_dict

    def get_atom_distance_matrix(self, pdb_array):
        coord_array = np.empty((pdb_array.shape[0], 3))
        for i, res_array in enumerate(pdb_array):
            coord_array[i] = res_array[7:10].astype(np.float64)
        atom_dm = squareform(pdist(coord_array))
        return coord_array, atom_dm

    def get_node_mfs_pos(self, pdb_array, residue_index):
        coord_array = np.empty((residue_index.shape[0], 3))
        for i in range(residue_index.shape[0]):
            # 获取每一个氨基酸中α碳原子的位置，若该氨基酸中有α碳原子，则取其位置，否则，则取该氨基酸中所有原子的平均值
            res_start, res_end = residue_index[i]
            flag = False
            res_array = pdb_array[res_start:res_end + 1]
            for j in range(res_array.shape[0]):
                if res_array[j][2] == 'CA':
                    coord_array[i] = res_array[:, 7:10][j].astype(np.float64)
                    flag = True
                    break
            if not flag:
                coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
                coord_array[i] = np.mean(coord_i, axis=0)
        return coord_array

    def get_atom_mfs_pos(self, pdb_array):
        coord_array = np.empty((pdb_array.shape[0], 3))
        for i, res_array in enumerate(pdb_array):
            coord_array[i] = res_array[7:10].astype(np.float64)
        return coord_array

    def get_residue_distance_matrix(self, pdb_array, residue_index, distance_type, mut_pos, wild_aa3):
        mut_pos_index = 0
        if distance_type == 'c_alpha':
            coord_array = np.empty((residue_index.shape[0], 3))
            for i in range(residue_index.shape[0]):
                # 获取每一个氨基酸中α碳原子的位置，若该氨基酸中有α碳原子，则取其位置，否则，则取该氨基酸中所有原子的平均值
                res_start, res_end = residue_index[i]
                flag = False
                res_array = pdb_array[res_start:res_end + 1]
                for j in range(res_array.shape[0]):
                    if res_array[j][-1] in metal_List:
                        metal_pos = i
                    if res_array[j][2] == 'CA':
                        coord_array[i] = res_array[:, 7:10][j].astype(np.float64)
                        flag = True
                        break
                if not flag:
                    coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
                    coord_array[i] = np.mean(coord_i, axis=0)
            residue_dm = squareform(pdist(coord_array))
        elif distance_type == 'mfs':
            zubiao_list = []
            residue_dm = np.empty((residue_index.shape[0],residue_index.shape[0]))
            for i in range(residue_index.shape[0]):
                res_start,res_end = residue_index[i]
                coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
                zubiao_list.append(np.array(coord_i))
                res_array = pdb_array[res_start:res_end + 1]
                for j in range(res_array.shape[0]):
                    if res_array[j][6] == mut_pos and res_array[j][4] == wild_aa3:
                        mut_pos_index = i
            for i in range(len(zubiao_list)):
                for j in range(len(zubiao_list)):
                    if i != j:
                        distance = cdist(zubiao_list[i], zubiao_list[j], 'euclidean')
                        residue_dm[i][j] = np.min(distance)
                    else:
                        residue_dm[i][j] = 0
        elif distance_type == 'centroid':
            coord_array = np.empty((residue_index.shape[0], 3))
            for i in range(residue_index.shape[0]):
                res_start, res_end = residue_index[i]
                coord_i = pdb_array[:, 7:10][res_start:res_end + 1].astype(np.float64)
                coord_array[i] = np.mean(coord_i, axis=0)
            residue_dm = squareform(pdist(coord_array))
        elif distance_type == 'atoms_average':
            full_atom_dist = squareform(pdist(pdb_array[:, 7:10].astype(float)))
            residue_dm = np.zeros((residue_index.shape[0], residue_index.shape[0]))
            for i, j in itertools.combinations(range(residue_index.shape[0]), 2):
                index_i = residue_index[i]
                index_j = residue_index[j]
                distance_ij = np.mean(full_atom_dist[index_i[0]:index_i[1] + 1, index_j[0]:index_j[1] + 1])
                residue_dm[i][j] = distance_ij
                residue_dm[j][i] = distance_ij
        else:
            raise ValueError('Invalid distance type: %s' % distance_type)
        return residue_dm, mut_pos_index

    def get_atom_neighbor_index(self, atom_dm, threshold=3):
        source, target, distance = [], [], []
        for i in range(atom_dm.shape[0]):
            for j in range(atom_dm.shape[1]):
                if atom_dm[i, j] <= threshold and atom_dm[i, j] != 0 :
                    source.append(i)
                    target.append(j)
                    distance.append(atom_dm[i, j])
        return source, target, distance

    def vector_dot(self, digit):
        if digit > 1:
            digit = 1
        elif digit < -1:
            digit = -1
        return digit

    def get_atom_neighbor_angle(self, coord_array, source, target):
        polar_angle, azimuthal_angle = [], []
        for (pos1, pos2) in zip(source, target):
            if pos1 == pos2:
                polar_angle.append(0.0)
                azimuthal_angle.append(0.0)
            else:
                coord_array_1 = coord_array[pos1]
                coord_array_2 = coord_array[pos2]
                direction_vector = coord_array_1 - coord_array_2
                unit_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                direction_vector[2] = 0.0
                projection_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                z_axis = np.array([0.0, 0.0, 1.0])
                x_axis = np.array([1.0, 0.0, 0.0])
                p_angle = np.arccos(get_edge().vector_dot(np.sum(unit_vector * z_axis))) / np.pi
                a_angle = np.arccos(get_edge().vector_dot(np.sum(projection_vector * x_axis))) / np.pi
                if p_angle!=p_angle: p_angle = np.float64(0.0)
                if a_angle!=a_angle: a_angle = np.float64(0.0)
                polar_angle.append(p_angle)
                azimuthal_angle.append(a_angle)
        return polar_angle, azimuthal_angle


    def get_residue_neighbor_angle_2(self, pdb_array, residue_index,source, target):
        coord_array = get_edge().get_node_mfs_pos(pdb_array, residue_index)
        polar_angle, azimuthal_angle = [], []
        for (pos1, pos2) in zip(source, target):
            if pos1 == pos2:
                polar_angle.append(0.0)
                azimuthal_angle.append(0.0)
            else:
                coord_array_1 = coord_array[pos1]
                coord_array_2 = coord_array[pos2]
                direction_vector = coord_array_1 - coord_array_2
                unit_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                direction_vector[2] = 0.0
                projection_vector = (direction_vector) / np.linalg.norm(direction_vector, 2)
                z_axis = np.array([0.0, 0.0, 1.0])
                x_axis = np.array([1.0, 0.0, 0.0])
                p_angle = np.arccos(get_edge().vector_dot(np.sum(unit_vector * z_axis))) / np.pi
                a_angle = np.arccos(get_edge().vector_dot(np.sum(projection_vector * x_axis))) / np.pi
                if p_angle != p_angle: p_angle = np.float64(0.0)
                if a_angle != a_angle: a_angle = np.float64(0.0)
                polar_angle.append(p_angle)
                azimuthal_angle.append(a_angle)
        return  polar_angle, azimuthal_angle

    def get_normal(self, acid_plane):
        cp = np.cross(acid_plane[2] - acid_plane[1], acid_plane[0] - acid_plane[1])
        if np.all(cp == 0):
            return np.array([np.nan] * 3)
        normal = cp / np.linalg.norm(cp, 2)
        return normal

    def fill_nan_mean(self, array, axis=0):
        if axis not in [0, 1]:
            raise ValueError('Invalid axis: %s' % axis)
        mean_array = np.nanmean(array, axis=axis)
        inds = np.where(np.isnan(array))
        array[inds] = np.take(mean_array, inds[1 - axis])
        if np.any(np.isnan(array)):
            full_array_mean = np.nanmean(array)
            inds = np.unique(np.where(np.isnan(array))[1 - axis])
            if axis == 0:
                array[:, inds] = full_array_mean
            else:
                array[inds] = full_array_mean
        return array

    def get_residue_edge_data(self, residue_dm, neighbor_index, neighbor_angle):
        edge_matrix = np.zeros((neighbor_index.shape[0], neighbor_index.shape[1], 2))
        for i, dist in enumerate(residue_dm):
            edge_matrix[i][:, 0] = dist[neighbor_index[i]]
            edge_matrix[i][:, 1] = neighbor_angle[i]
        return edge_matrix

    def get_neighbor_index(self, residue_dm):
        return residue_dm.argsort()[:, :]

    def add_pydca(self, edge_index, edge_feature, plmdca_dict, mfdca_dict, res_index_pos_dict_premut, pdb2uniprot_pos):
        edge_feature = edge_feature.tolist()
        for (i, pos) in enumerate(zip(edge_index[0], edge_index[1])):
            uniprot_pos1 = pdb2uniprot_pos[res_index_pos_dict_premut[pos[0]]]
            uniprot_pos2 = pdb2uniprot_pos[res_index_pos_dict_premut[pos[1]]]
            if uniprot_pos1 == uniprot_pos2:
                edge_feature[i].append(0.00)
                edge_feature[i].append(0.00)
                continue
            else:
                if int(uniprot_pos1) < int(uniprot_pos2): index = uniprot_pos1+'_'+uniprot_pos2
                else: index = uniprot_pos2+'_'+uniprot_pos1
                plmdca, mfdca = plmdca_dict[index], mfdca_dict[index]
                edge_feature[i].append(float(plmdca))
                edge_feature[i].append(float(mfdca))
        # print(edge_feature)
        edge_feature = np.array(edge_feature)
        return edge_feature


    def generate_res_edge_feature_postmut(self, mut_pos, pdb_array, distance_type='c_alpha'):
        residue_index, pdb_pos_list = get_edge().get_residue_info(pdb_array)
        residue_dm = get_edge().get_residue_distance_matrix(pdb_array, residue_index, distance_type)
        neighbor_index = get_edge().get_neighbor_index(residue_dm)
        neighbor_angle = get_edge().get_residue_neighbor_angle(pdb_array, residue_index, neighbor_index)

        mut_index = pdb_pos_list.index(mut_pos)
        edge_num = len(pdb_pos_list)
        source = np.array([mut_index for i in range(edge_num)])
        target = np.array([i for i in range(edge_num)])

        edge_data = np.empty((len(pdb_pos_list), 2), dtype=np.float64)
        for s, t in zip(source, target):
            edge_data[t][0] = residue_dm[s][t]
            edge_data[t][1] = neighbor_angle[s][t]

        edge_index = [source, target]
        return edge_index, edge_data, residue_index

    def generate_atom_edge_feature(self, pdb_array):
        coord_array, atom_dm = get_edge().get_atom_distance_matrix(pdb_array)
        source, target, distance = get_edge().get_atom_neighbor_index(atom_dm, 3)
        polar_angle, azimuthal_angle = get_edge().get_atom_neighbor_angle(coord_array, source, target)
        edge_feature = np.empty((len(source), 3))
        for (i, edge_info) in enumerate(zip(distance, polar_angle, azimuthal_angle)):
            edge_feature[i][0] = edge_info[0]
            edge_feature[i][1] = edge_info[1]
            edge_feature[i][2] = edge_info[2]
        edge_index = [np.array(source), np.array(target)]
        return edge_index, edge_feature

    def get_mfs_redisue_neighbor_index(self, residue_dm,mut_pos_index, threshold1=3, threshold2=5):
        source, target, distance,ligand1,ligand2 = [], [], [], [], []
        for i in range(residue_dm.shape[0]):
            if i!=mut_pos_index and residue_dm[mut_pos_index, i] <= threshold1:
                source.append(mut_pos_index)
                target.append(i)
                distance.append(residue_dm[mut_pos_index, i])
                ligand1.append(i)
        for lig1_pos in ligand1:
            for i in range(residue_dm.shape[0]):
                if i not in ligand1 and i!=mut_pos_index and residue_dm[lig1_pos, i] <= threshold2:
                    source.append(lig1_pos)
                    target.append(i)
                    distance.append(residue_dm[lig1_pos, i])
                    ligand2.append(i)
        return source, target, distance


    def get_new_pdb_array(self,pdb_array,find_res_pos):
        new_pdb_array = []
        res_index_pos_dict, atom_index_pos, pos_index_resname = {}, {}, {}
        res_pos = -10
        res_index, atom_index = 0, 0
        for atom_info in pdb_array:
            if (atom_info[6] in find_res_pos):
                new_pdb_array.append(atom_info)
                atom_index_pos[atom_index] = atom_info[1]
                atom_index += 1
                pos_index_resname[atom_info[6]] = atom_info[4]
                if atom_info[6] != res_pos:
                    res_index_pos_dict[res_index] = atom_info[6]
                    res_index += 1
                    res_pos = atom_info[6]
        return np.array(new_pdb_array, dtype='str'), res_index_pos_dict, atom_index_pos, pos_index_resname

    # 根据MFS修改后的，也就是把金属离子3埃之内的化合物作为第一配位，距离第一配位5埃之内的化合物作为第二配位
    def generate_mfs_residue_edge_feature(self, pdb_array, pdb_tag, pdbdir_path, mut_pos, wild_aa3, distance_type='mfs'):
        residue_index, pdb_pos_dict = get_edge().get_residue_info(pdb_array)
        residue_dm, mut_pos_index = get_edge().get_residue_distance_matrix(pdb_array, residue_index, distance_type, mut_pos, wild_aa3)
        #每个氨基酸残基到其他残基的距离排序
        source, target, distance = get_edge().get_mfs_redisue_neighbor_index(residue_dm,mut_pos_index,3,5)
        find_res = list(set(list(set(source))+list(set(target))))
        find_res_pos = []
        for key in pdb_pos_dict.keys():
            if key in find_res:
                find_res_pos.append(pdb_pos_dict[key])
        #清洗掉没有找到的res
        new_pdb_array, res_index_pos_dict_wt, atom_index_pos_wt, pos_index_resname_wt = get_edge().get_new_pdb_array(pdb_array,find_res_pos)

        #重新映射
        residue_index, pdb_pos_dict = get_edge().get_residue_info(new_pdb_array)
        residue_dm, mut_pos_index = get_edge().get_residue_distance_matrix(new_pdb_array, residue_index, distance_type, mut_pos, wild_aa3)
        # 每个氨基酸残基到其他残基的距离排序
        source, target, distance = get_edge().get_mfs_redisue_neighbor_index(residue_dm, mut_pos_index, 3, 5)

        polar_angle, azimuthal_angle = get_edge().get_residue_neighbor_angle_2(pdb_array, residue_index, source, target)

        is_Hbond_list = []
        donor, acceptor = get_Hbond(pdb_tag,pdbdir_path)
        for i, item_s in enumerate(source):
            is_Hbond = 0
            for Hbond in zip(donor, acceptor):
                if (pdb_pos_dict[item_s] == Hbond[0] and pdb_pos_dict[target[i]] == Hbond[1]) or (pdb_pos_dict[item_s] == Hbond[1] and pdb_pos_dict[target[i]] == Hbond[0]):
                    is_Hbond = 1
                    break
            is_Hbond_list.append(is_Hbond)
        basic_attn = np.empty((len(source), 1))
        basic_attn_peiwei = np.empty((len(source), 1))
        for (i,attn_item) in enumerate(zip(source,target,distance)):
            fenzi = 1
            fenzi_pw = 1
            c = 0.2
            fenmu = math.exp(attn_item[2]*c)
            pdbpos_s = pdb_pos_dict[attn_item[0]]
            pdbpos_t = pdb_pos_dict[attn_item[1]]
            resname_s = pos_index_resname_wt[pdbpos_s]
            resname_t = pos_index_resname_wt[pdbpos_t]
            if (resname_s in zheng_res and resname_t in fu_res) or (resname_s in fu_res and resname_t in zheng_res):
                fenzi += 1
                fenzi_pw += 1
            if resname_s in HYDROPHOBIC_res and resname_t in HYDROPHOBIC_res:
                fenzi += 1
                fenzi_pw += 1
            if is_Hbond_list[i]==1 and resname_s not in metal_List and resname_t not in metal_List:
                fenzi += 2
                fenzi_pw += 2
            if resname_s in metal_List or  resname_t in metal_List:
                fenzi_pw += 2
            basic_attn[i][0] = float(fenzi/fenmu)
            basic_attn_peiwei[i][0] =  float(fenzi_pw/fenmu)
        edge_index = [np.array(source), np.array(target)]
        edge_feature = np.empty((len(source), 4))
        for (i, edge_info) in enumerate(zip(distance, polar_angle, azimuthal_angle, is_Hbond_list)):
            edge_feature[i][0] = edge_info[0]
            edge_feature[i][1] = edge_info[1]
            edge_feature[i][2] = edge_info[2]
            edge_feature[i][3] = edge_info[3]
        node_pos = get_edge().get_node_mfs_pos(new_pdb_array, residue_index)
        atom_pos = get_edge().get_atom_mfs_pos(new_pdb_array)
        node_pos_dict = {}
        atom_pos_dict = {}
        for i,node_pos_array in enumerate(node_pos):
            node_pos_dict[res_index_pos_dict_wt[i]] = node_pos_array.tolist()
        for i,atom_pos_array in enumerate(atom_pos):
            atom_pos_dict[atom_index_pos_wt[i]] = atom_pos_array.tolist()
        return new_pdb_array,edge_index, edge_feature, residue_index,node_pos,res_index_pos_dict_wt, atom_index_pos_wt,node_pos_dict,atom_pos_dict,basic_attn,basic_attn_peiwei





