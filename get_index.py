import os
import pickle
import random
import torch
import shutil

all_name = []

def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def get_unique_prefixes(directory):
    from collections import defaultdict
    # 创建一个字典，键为组合，值为文件名列表
    combination_to_files = defaultdict(list)
    # 从文本文件中读取所有文件名
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filename = filename.split('.')[0]
            all_name.append(filename)
            parts = filename.split('_')
            combination = tuple(parts[i] for i in [0, 1, 3, 4, 6, 7] if i < len(parts))
            # 将文件名添加到对应的组合列表中
            combination_to_files[combination].append(filename)

    prefixes = []
    for files in combination_to_files.values():
        if files:  # 确保列表不为空
            min_file = min(files, key=lambda f: int(f.split('_')[2]))
            prefixes.append(min_file)
    return prefixes


def save_prefixes_to_pkl(prefixes, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(prefixes, f)


directory = 'Data/MPD476/seq_to_fea'

# 获取唯一文件名前缀集合
prefixes = get_unique_prefixes(directory)
prefixes = list(prefixes)


i = 0
label_list = []
ddg_list = []
for item in prefixes:
    # 定义PKL文件的路径
    label_file_path = 'Data/MPD476/label/' + item + '.pt'
    label = torch.load(label_file_path)
    label_list.append(label)
    ddg_file_path = 'Data/MPD476/ddg/' + item + '.pt'
    ddg = torch.load(ddg_file_path)
    ddg_list.append(ddg)

data_index_folder = "Data/MPD476/data_index"
RemoveDir(data_index_folder)

# 保存前缀集合到pkl文件
save_prefixes_to_pkl(prefixes, data_index_folder+'/MPD476.pkl')
save_prefixes_to_pkl(label_list, data_index_folder+'/MPD476_label.pkl')
save_prefixes_to_pkl(ddg_list, data_index_folder+'/MPD476_ddg.pkl')


