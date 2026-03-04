# -- coding: utf-8 --
import os
import pickle
import random
import re
import shutil
from urllib import request

import math
import numpy
import numpy as np
import pandas as pd
import torch
import wget
from Bio import PDB
from propy import AAComposition as AAC
from propy import Autocorrelation as AC
from propy import CTD as CTD
from propy import QuasiSequenceOrder as QSO
from propy import ProCheck as PC
from scipy.sparse import coo_matrix
from selenium import webdriver
import requests

numpy.random.seed(42)
random.seed(42)


def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def Removefile(file):
    if os.path.exists(file):
        os.remove(file)


def change_three_to_one(res):
    amino_acid = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PRO': 'P', 'PHE': 'F', 'TYR': 'Y',
                  'TRP': 'W', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'ASN': 'N', 'GLN': 'Q', 'ASP': 'D',
                  'GLU': 'E', 'LYS': 'K', 'ARG': 'R', 'HIS': 'H',
                  }
    if res in amino_acid:
        res = amino_acid[res]
        return res
    else:
        return '_'


def clean_pdb(pdb_file):
    out_pdb_file = temp_pdb_filder + '/cleanafter.pdb'
    with open(pdb_file,'r') as f_r,open(out_pdb_file,'w') as f_w:
        for line in f_r:
            info = [line[0:5], line[6:11], line[12:16], line[16], line[17:20], line[21], line[22:27], line[30:38],
                    line[38:46], line[46:54]]
            info = [i.strip() for i in info]
            if info[0] == 'ATOM' or info[0] == 'HETAT':
                f_w.write(line)
            if 'ENDMDL' in line:
                break


def change_one_to_three(res):
    three_letter = {'V': 'VAL', 'I': 'ILE', 'L': 'LEU', 'E': 'GLU', 'Q': 'GLN',
                    'D': 'ASP', 'N': 'ASN', 'H': 'HIS', 'W': 'TRP', 'F': 'PHE', 'Y': 'TYR',
                    'R': 'ARG', 'K': 'LYS', 'S': 'SER', 'T': 'THR', 'M': 'MET', 'A': 'ALA',
                    'G': 'GLY', 'P': 'PRO', 'C': 'CYS'}
    if res in three_letter:
        res = three_letter[res]
        return res
    else:
        return '_'


# 20种氨基酸
resList = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
           "SER", "THR", "TRP", "TYR", "VAL"]
pdb_list = []


def get_pipeipdb_info(pdbname, chain_name, nucleic_acid, pos, from_acid, to_acid, ddg):
    global set_to_acid
    RemoveDir('../Data/demo')
    tag = False

    for i in range(1):
        toX_tag = False
        if to_acid == 'X':
            # 指定要排除的元素
            excluded_value = change_one_to_three(from_acid)
            # 创建一个新的列表，不包含指定的元素
            filtered_list = [x for x in resList if x != excluded_value]
            to_acid = change_three_to_one(random.choice(filtered_list))
            set_to_acid = to_acid
            toX_tag = True
        pdbfile = "../Data/PDB/RNA/{}.pdb".format(pdbname)
        if os.path.getsize(pdbfile) == 0:
            break
        parser = PDB.PDBParser(QUIET=True)
        struct = parser.get_structure('PDB', pdbfile)
        model = struct[0]
        flag = False
        from_seq = list()
        to_seq = list()
        len = 0
        for chain in model:
            for res in chain:
                # if not flag: cow_pos += 1
                amino_acid = res.get_resname()
                res_id = res.get_id()
                orig_pos = str(res_id[1]).strip() + str(res_id[2]).strip()
                one_acid = change_three_to_one(amino_acid)
                if one_acid != '_':
                    from_seq.append(one_acid)
                    to_seq.append(one_acid)
                    len += 1
                if one_acid == from_acid and orig_pos == str(pos).strip():
                    to_seq.pop()
                    to_seq.append(to_acid)
                    flag = True
        pocketSeq = "".join(from_seq)
        pocketSeq_to = "".join(to_seq)
        if toX_tag == True:
            to_acid = 'X'
        if flag == True:
            # foldx得到突变后的pdb
            # pdb = pdbfile.split('/')[-1]
            clean_pdb(pdbfile)
            pdb = temp_pdb_filder + '/cleanafter.pdb'
            # pdb = 'Data/foldx/cleanafter.pdb'
            if not os.path.exists(folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(nucleic_acid,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)):
                os.rename(pdb,folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(nucleic_acid,pdbname.lower(),i,pos,from_acid,chain_name,to_acid))
            end_pdb = folder_from+'/{}_{}_{}_{}_{}_{}_{}.pdb'.format(nucleic_acid,pdbname.lower(),i,pos,from_acid,chain_name,to_acid)
            tag = True

            nucleic_acid_fea = [nucleic_acid, from_acid, to_acid]
            nucleic_acid_fea = np.array(nucleic_acid_fea).reshape(1, -1)


            from sklearn.preprocessing import LabelEncoder, OneHotEncoder
            onehotencoder = OneHotEncoder(sparse_output=False,
                                          categories=[nucleic_acid_List, from_acid_list, to_acid_list])
            onehotencoder_res = OneHotEncoder(sparse_output=False, categories=[to_acid_list])
            nucleic_acid_fea = onehotencoder.fit_transform(nucleic_acid_fea)


            if ddg > 1 or ddg < -1:
                label = 1
            elif ddg <= 1 and ddg >=-1:
                label = 0

            y_ddg = np.array(ddg).astype(float)
            y_ddg = torch.tensor(y_ddg, dtype=torch.float)
            y = np.array(label).astype(int)
            y = torch.tensor(y, dtype=torch.int64)
            nucleic_acid_fea = np.array(nucleic_acid_fea).flatten().astype(float)
            nucleic_acid_fea = torch.tensor(nucleic_acid_fea, dtype=torch.float)
            print(pdbname, y_ddg, y)

            torch.save(y_ddg, ddg_path_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(nucleic_acid, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            torch.save(y, labelpath_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(nucleic_acid, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            torch.save(nucleic_acid_fea, metal_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(nucleic_acid, pdbname.lower(), i, pos, from_acid, chain_name,to_acid))

            all_pdb_tag.append('{}_{}_{}_{}_{}_{}_{}'.format(nucleic_acid, pdbname.lower(),i,pos,from_acid,chain_name,to_acid))

            ProteinSequence = pocketSeq
            ProteinSequence_to = pocketSeq_to
            try:
                if (PC.ProteinCheck(ProteinSequence) > 0):
                    # print "Protein A Composition (Percent) - 20"
                    # 举例 A=7/39*100 保留三位小数
                    dicAAC = AAC.CalculateAAComposition(ProteinSequence)
                    dicAAC_to = AAC.CalculateAAComposition(ProteinSequence_to)

                    # print "Protein CTD - All - 147"
                    dicCTD = CTD.CalculateCTD(ProteinSequence)
                    dicCTD_to = CTD.CalculateCTD(ProteinSequence_to)

                    # print "Protein Autocorrelation - All - 720"
                    dicAC = AC.CalculateAutoTotal(ProteinSequence)
                    dicAC_to = AC.CalculateAutoTotal(ProteinSequence_to)

                    # print "Protein Quasi-sequence order descriptors - 100"
                    dicQSO = QSO.GetQuasiSequenceOrder(ProteinSequence)
                    dicQSO_to = QSO.GetQuasiSequenceOrder(ProteinSequence_to)

                    # print "Protein Sequence order coupling number descriptors - 60"
                    dicQSO2 = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence)
                    dicQSO2_to = QSO.GetSequenceOrderCouplingNumberTotal(ProteinSequence_to)

                    # dicPAAC = PAAC.CalculatePAAComposition(ProteinSequence)
                    dicAll = dict(
                        list(dicAAC.items()) + list(dicCTD.items()) + list(dicAC.items()) + list(dicQSO.items()) + list(
                            dicQSO2.items()))
                    dicAll_to = dict(
                        list(dicAAC_to.items()) + list(dicCTD_to.items()) + list(dicAC_to.items()) + list(
                            dicQSO_to.items()) + list(
                            dicQSO2_to.items()))

                    # featureNames = list(dicAll.keys())
                    seqfeatures = list(dicAll.values())
                    seqfeatures_to = list(dicAll_to.values())
                    seqfeatures.insert(0,pos)
                    seqfeatures_to.insert(0,pos)

                    seqfeatures = np.array(seqfeatures).astype(float)
                    seqfeatures = torch.tensor(seqfeatures, dtype=torch.float)
                    torch.save(seqfeatures,seq_from_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(nucleic_acid, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))

                    seqfeatures_to = np.array(seqfeatures_to).astype(float)
                    seqfeatures_to = torch.tensor(seqfeatures_to, dtype=torch.float)
                    torch.save(seqfeatures_to,seq_to_filder + '/{}_{}_{}_{}_{}_{}_{}.pt'.format(nucleic_acid, pdbname.lower(), i, pos, from_acid,chain_name, to_acid))
            except:
                print("********pdb:{},nucleic_acid:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, nucleic_acid, pos, from_acid,to_acid))
                with open(error_examples_file, 'a') as f:
                    f.write("********pdb:{},nucleic_acid:{},pos:{},fromacid:{},toacid:{},xulie_error".format(pdbname, nucleic_acid, pos, from_acid,to_acid) + '\n')

    if not tag:
        print("未找到pdb:{},nucleic_acid:{},pos:{},acid:{},toacid:{}样本对应的氨基酸".format(pdbname, nucleic_acid, pos, from_acid,to_acid))
        with open(error_examples_file, 'a') as f:
            f.write("未找到pdb:{},nucleic_acid:{},pos:{},acid:{},toacid:{}样本对应的氨基酸".format(pdbname, nucleic_acid, pos, from_acid,to_acid) + '\n')
    return


def get_excel_data(file):
    df = pd.read_excel(file)

    pattern = r'^([A-Za-z])(\d+)([A-Za-z])$'
    mutations = df['mutation_old'].str.extract(pattern)
    orig_acid = mutations[0]
    location = mutations[1]
    to_acid = mutations[2]

    nucleic_acid = df['Nucleic_Acid']
    pdbname = df['pdb_id']
    chain = df['chain']
    ddg = df['ddg']

    return nucleic_acid.values, pdbname.values, chain.values, location.values, orig_acid.values, to_acid.values, ddg.values


def get_metal_seq_info(excel_add):
    nucleic_acid, pdbname, chain, location, orig_acid, to_acid, ddg = get_excel_data(excel_add)
    for i in range(0, len(nucleic_acid)):
        if pd.isna(location[i]):continue
        get_pipeipdb_info(pdbname[i], chain[i], nucleic_acid[i], int(location[i]), orig_acid[i], to_acid[i], ddg[i])



from_acid_list = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H']
to_acid_list = ['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', 'X']
nucleic_acid_List = ["DNA", "RNA"]

if __name__ == '__main__':

    folder_from = '../Data/MPD476/nucleic_acid_pdb'
    seq_from_filder = "../Data/MPD476/seq_fea"
    seq_to_filder = "../Data/MPD476/seq_to_fea"
    metal_filder = "../Data/MPD476/nucleic_acid_fea"
    labelpath_filder = "../Data/MPD476/label"
    ddg_path_filder = "../Data/MPD476/ddg"
    data_path = '../Data/MPD476/record_data.txt'
    error_examples_file = "../Data/MPD476/nofind_pdb.txt"
    temp_pdb_filder = "../Data/temp_pdb"

    Removefile(error_examples_file)
    Removefile(data_path)
    RemoveDir(folder_from)
    RemoveDir(seq_from_filder)
    RemoveDir(seq_to_filder)
    RemoveDir(metal_filder)
    RemoveDir(labelpath_filder)
    RemoveDir(ddg_path_filder)
    RemoveDir(temp_pdb_filder)

    all_pdb_tag = []
    get_metal_seq_info('../Data/MPD476/dna_MPD476.xlsx')
    # all_pdb_tag = list(set(all_pdb_tag))
    with open(data_path, 'w') as f:
        for item in all_pdb_tag:
            f.write("%s\n" % item)

