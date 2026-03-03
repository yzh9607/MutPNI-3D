import os
import subprocess
from sys import stdout, stderr

sasa_path = './Data/SASA'
if not os.path.exists(sasa_path):
    os.makedirs(sasa_path)

def naccess(pdb_file):
    res_naccess_output, atom_naccess_output = [], []
    pdb_name = os.path.basename(pdb_file).split('.')[0]
    if not os.path.exists(os.path.join(sasa_path, pdb_name + '.rsa')):
        ##naccess压缩包请去官网下载，按照Readme进行解压，路径填你自己环境中naccess的路径
        # 获取文件的绝对路径
        file_path = os.path.abspath(pdb_file)
        original_dir = os.getcwd()
        os.chdir(sasa_path)
        pdb_file = pdb_file.replace('../Data/', '../')
        stdout, stderr = subprocess.Popen(['/home/yanzihao/bio_tool/naccess2.1.1/naccess', file_path, '-h'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True).communicate()
        print("Standard Output:", stdout)
        print("Standard Error:", stderr)
        os.chdir(original_dir)
        #os.system(f'mv {sasa_path}/5.rsa {pdb_name}.rsa')
        #os.system(f'mv {sasa_path}/5.asa {pdb_name}.asa')
    try:
        res_naccess_output += open(os.path.join(sasa_path, pdb_name + '.rsa'), 'r').readlines()
        atom_naccess_output += open(os.path.join(sasa_path, pdb_name + '.asa'), 'r').readlines()

    except IOError:
        raise IOError('ERROR: Naccess .rsa file was not written. The following command was attempted: %s %s' % (
            'naccess', pdb_file))

    return res_naccess_output, atom_naccess_output


def SASA(pdb_file):
    res_sasa_dict, atom_sasa_dict= {}, {}
    res_naccess_output, atom_naccess_output = naccess(pdb_file)
    # print(res_naccess_output, atom_naccess_output)
    for res_info in res_naccess_output:
        if res_info[0:3] == 'RES' or res_info[0:3] == 'HEM':
            residue_index = res_info[9:14].strip()
            relative_perc_accessible = float(res_info[22:28])
            res_sasa_dict[residue_index] = relative_perc_accessible

    # 注意原子的位置也要对应起来
    for atom_info in atom_naccess_output:
        # atom_info = atom_info.split()
        if atom_info[0:4] == 'ATOM':
            atom_index = atom_info[6:11].strip()
            relative_perc_accessible = atom_info[54:62].strip()
            atom_sasa_dict[atom_index] = relative_perc_accessible
        if atom_info[0:4] == 'HETA':
            hem_index = atom_info[7:12].strip()
            relative_perc_accessible = atom_info[55:63].strip()
            atom_sasa_dict[hem_index] = relative_perc_accessible

    # print(res_sasa_dict, atom_sasa_dict)
    return res_sasa_dict, atom_sasa_dict
