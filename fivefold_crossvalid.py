# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from numpy import mean, argsort
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, \
    matthews_corrcoef, f1_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit
from torch import nn
from torch.nn import BCELoss
from torch.utils.data import DataLoader
# from CASTLE.Model.model import Model_Net
from Model.model import Model_Net
import Dataset as dataset
import openpyxl as op
from scipy.stats import pearsonr, spearmanr


def augment_features(
        inputPT, foldx,
        noise_level=0.01, swap_prob=0.05, scale_range=(0.98, 1.02)
):
    """
    数据增强：对不同特征做轻量级扰动（仅训练时使用）
    """
    def add_gaussian_noise(x):
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def scale_feature(x):
        scale = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).to(x.device)
        return x * scale

    inputPT_aug = add_gaussian_noise(inputPT)
    inputPT_aug = scale_feature(inputPT_aug)
    foldx_aug = scale_feature(foldx)

    return inputPT_aug, foldx_aug


def op_toexcel(data,filename):

    if os.path.exists(filename):
        wb = op.load_workbook(filename)
        ws = wb.worksheets[0]

        ws.append(data)
        wb.save(filename)
    else:
        wb = op.Workbook()
        ws = wb['Sheet']
        ws.append(['PCC', 'RMSE', 'MAE', 'MSE', 'R²', 'ρ', '斯皮尔曼p值', 'MedAE', 'MaxAE', '皮尔逊p值'])
        ws.append(data)
        wb.save(filename)

def fcvtest_regression(test_pred, test_label, output_result):
    y_pred = np.array(test_pred).ravel()  # 展平为一维数组，避免维度问题
    y_true = np.array(test_label).ravel()

    mae = metrics.mean_absolute_error(y_true, y_pred)  # 平均绝对误差
    mse = metrics.mean_squared_error(y_true, y_pred)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差

    r2 = metrics.r2_score(y_true, y_pred)  # 决定系数（R²）
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)  # 皮尔逊相关系数（线性相关）
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)  # 斯皮尔曼相关系数（非线性/单调相关）

    med_ae = metrics.median_absolute_error(y_true, y_pred)  # 中位数绝对误差
    max_ae = np.max(np.abs(y_true - y_pred))  # 最大绝对误差

    print("=" * 50 + " 回归任务评估结果 " + "=" * 50)
    print(f"皮尔逊相关系数 (r): {pearson_corr:.4f} ")
    print(f"RMSE（均方根误差）: {rmse:.4f}")
    print(f"MAE（平均绝对误差）: {mae:.4f}")
    print(f"MSE（均方误差）: {mse:.4f}")
    # print(f"MAPE（平均绝对百分比误差）: {mape:.4f}%")
    print(f"R²（决定系数）: {r2:.4f}")
    print(f"斯皮尔曼相关系数 (ρ): {spearman_corr:.4f} (p值: {spearman_p:.4f})")
    print(f"MedAE（中位数绝对误差）: {med_ae:.4f}")
    print(f"MaxAE（最大绝对误差）: {max_ae:.4f}")
    print(f"皮尔逊p值: {pearson_p:.4f}")

    result = (
        float(format(pearson_corr, '.4f')),
        float(format(rmse, '.4f')),
        float(format(mae, '.4f')),
        float(format(mse, '.4f')),
        float(format(r2, '.4f')),
        float(format(spearman_corr, '.4f')),
        float(format(spearman_p, '.4f')),
        float(format(med_ae, '.4f')),
        float(format(max_ae, '.4f')),
        float(format(pearson_p, '.4f'))
    )
    op_toexcel(result, output_result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--data-index', dest='data_index', type=str,default='Data/MPD476/data_index/MPD476.pkl')
    parser.add_argument('-il', '--data-label-index', dest='data_label_index', type=str,default='Data/MPD476/data_index/MPD476_ddg.pkl')
    parser.add_argument('-or', '--outputdir-result', dest='outputdir_result', type=str,default=f'Result/all_indicator/5fold_result.xlsx')
    parser.add_argument('-of', '--outputdir-file', dest='outputdir_file', type=str,default=f'Result/all_file/5fold_result')

    args = parser.parse_args()
    data_index_file = args.data_index
    data_label_index_file = args.data_label_index
    output_result = args.outputdir_result
    outputdir_file = args.outputdir_file
    if os.path.exists(output_result):
        os.remove(output_result)  # 删除文件
    if not os.path.exists(outputdir_file):
        os.makedirs(outputdir_file)
    epochs = 500
    patience = 20
    lr = 1e-4
    batchsize = 1
    GPU_ID = 'cuda:1'
    devices = torch.device(GPU_ID if torch.cuda.is_available() else 'cpu')
    cv = KFold(n_splits=5)

    global fpr_keras
    global tpr_keras
    with open(data_index_file, 'rb') as f:
        all_pdb_tag = pickle.load(f)
    with open(data_label_index_file, 'rb') as f:
        all_label = pickle.load(f)
    all_pdb_tag = np.array(all_pdb_tag)
    all_label = np.array(all_label)
    fold_num = 0

    for train,test in cv.split(all_pdb_tag,all_label):
        fold_num += 1
        model = Model_Net()
        model.to(devices)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.85)

        pdb_tag_train = all_pdb_tag[train]
        pdb_label_train = all_label[train]
        pdb_tag_test = all_pdb_tag[test]
        data_test = dataset.mydataset(pdb_tag_test, 'Data/MPD476')
        data_loader_test = DataLoader(data_test, batch_size=batchsize, shuffle=True,drop_last=True,num_workers=0)

        split = ShuffleSplit(n_splits=1, test_size=0.2)
        best_model_wts = None
        for train_inx, valid_inx in split.split(pdb_tag_train, pdb_label_train):
            pdb_tag_train = all_pdb_tag[train_inx]
            pdb_tag_valid = all_pdb_tag[valid_inx]
            data_train = dataset.mydataset(pdb_tag_train, 'Data/MPD476')
            data_loader_train = DataLoader(data_train, batch_size=batchsize, shuffle=True, drop_last=True,
                                           num_workers=0)
            data_valid = dataset.mydataset(pdb_tag_valid, 'Data/MPD476')
            data_loader_valid = DataLoader(data_valid, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=0)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in range(epochs):
                one_epoch_y = []
                one_epoch_predic = []
                one_epoch_loss = 0

                count = 1
                model.train()
                for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, \
                        atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label, inputPT, foldx in data_loader_train:

                    optimizer.zero_grad()
                    seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, \
                        atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, inputPT, foldx = \
                        seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), res_index_from.to(devices), \
                            res_edge_from.to(devices), atom_node_from.to(devices), atom_index_from.to(devices), atom_edge_from.to(devices), \
                            atom2res_from.to(devices), node_pos_from.to(devices), basic_attn_from.to(devices), inputPT.to(devices), foldx.to(devices)

                    seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, \
                        atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, foldx = \
                        torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea, dim=0), torch.squeeze(res_node_from, dim=0), \
                            torch.squeeze(res_index_from, dim=0), torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), \
                            torch.squeeze(atom_index_from, dim=0), torch.squeeze(atom_edge_from, dim=0), torch.squeeze(atom2res_from, dim=0), \
                            torch.squeeze(node_pos_from, dim=0), torch.squeeze(basic_attn_from, dim=0), torch.squeeze(foldx, dim=0)

                    if model.training:  # 仅训练模式增强，验证/测试不增强
                        inputPT, foldx = augment_features(
                            inputPT, foldx,
                            noise_level=0.01,
                            swap_prob=0.05,
                            scale_range=(0.98, 1.02)
                        )

                    pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from, atom_node_from,
                                   atom_index_from, atom_edge_from, node_pos_from, basic_attn_from, seq_from, seq_to,
                                   metal_fea, devices, inputPT, foldx)

                    truth_y = label.to(devices).type(torch.float)
                    loss = criterion(truth_y,pred_y)
                    loss.to(devices)
                    out = pred_y.detach().cpu().numpy()
                    y = truth_y.detach().cpu().numpy()
                    one_epoch_predic.append(out.item())
                    one_epoch_y.append(y)
                    one_epoch_loss += loss.detach().cpu().numpy()
                    loss.backward()
                    count += 1
                    optimizer.step()

                print("第{}次epoch下训练集的Loss为{}".format(epoch, one_epoch_loss))
                scheduler.step()

                model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, \
                            atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label, inputPT, foldx in data_loader_valid:
                        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, \
                            atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, inputPT, foldx = \
                            seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), res_index_from.to(devices), \
                                res_edge_from.to(devices), atom_node_from.to(devices), atom_index_from.to(devices), atom_edge_from.to(devices), \
                                atom2res_from.to(devices), node_pos_from.to(devices), basic_attn_from.to(devices), inputPT.to(devices), foldx.to(devices)

                        seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, \
                            atom_index_from, atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, foldx = \
                            torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea,dim=0), torch.squeeze(res_node_from, dim=0), \
                                torch.squeeze(res_index_from, dim=0), torch.squeeze(res_edge_from,dim=0), torch.squeeze(atom_node_from, dim=0), \
                                torch.squeeze(atom_index_from, dim=0), torch.squeeze(atom_edge_from,dim=0), torch.squeeze(atom2res_from, dim=0), \
                                torch.squeeze(node_pos_from, dim=0), torch.squeeze(basic_attn_from,dim=0), torch.squeeze(foldx, dim=0)

                        pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from, atom_node_from,
                                       atom_index_from, atom_edge_from, node_pos_from, basic_attn_from, seq_from,
                                       seq_to, metal_fea, devices, inputPT, foldx)
                        label = label.to(devices).type(torch.float)
                        loss = criterion(label,pred_y)
                        val_loss += loss.item()

                        pred_flat = pred_y.detach().cpu().numpy().flatten()
                        label_flat = label.detach().cpu().numpy().flatten()
                        all_preds.extend(pred_flat)
                        all_labels.extend(label_flat)
                val_loss /= len(data_loader_valid)

                pcc, pcc_p_value = pearsonr(all_preds, all_labels)
                if np.isnan(pcc):
                    pcc = 0.0
                all_preds_np = np.array(all_preds)
                all_labels_np = np.array(all_labels)
                mse = np.mean((all_preds_np - all_labels_np) ** 2)
                rmse = np.sqrt(mse)
                print(f"第{epoch}次epoch下验证集的val_loss为：{val_loss:.6f}, PCC为{pcc:.6f}，RMSE为{rmse:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break



        test_predict = []
        test_y = []
        model.load_state_dict(best_model_wts)
        model.eval()
        df = pd.DataFrame(columns=['PDB', 'Chain_ID', 'FromAA', 'ToAA', 'Location', 'Label', 'Predict'])
        with torch.no_grad():
            for pdb_tag, metal_fea, seq_from, seq_to, res_node_from, res_edge_from, res_index_from, atom_node_from, atom_index_from, \
                    atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, label, inputPT, foldx in data_loader_test:
                seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, \
                    atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, inputPT, foldx = \
                    seq_from.to(devices), seq_to.to(devices), metal_fea.to(devices), res_node_from.to(devices), res_index_from.to(devices), \
                        res_edge_from.to(devices), atom_node_from.to(devices), atom_index_from.to(devices), atom_edge_from.to(devices), \
                        atom2res_from.to(devices), node_pos_from.to(devices), basic_attn_from.to(devices), inputPT.to(devices), foldx.to(devices)

                seq_from, seq_to, metal_fea, res_node_from, res_index_from, res_edge_from, atom_node_from, atom_index_from, \
                    atom_edge_from, atom2res_from, node_pos_from, basic_attn_from, foldx = \
                    torch.squeeze(seq_from, dim=0), torch.squeeze(seq_to, dim=0), torch.squeeze(metal_fea,dim=0), torch.squeeze(res_node_from, dim=0), \
                        torch.squeeze(res_index_from, dim=0), torch.squeeze(res_edge_from, dim=0), torch.squeeze(atom_node_from, dim=0), \
                        torch.squeeze(atom_index_from, dim=0), torch.squeeze(atom_edge_from, dim=0), torch.squeeze(atom2res_from, dim=0), \
                        torch.squeeze(node_pos_from, dim=0), torch.squeeze(basic_attn_from, dim=0), torch.squeeze(foldx,dim=0)

                pred_y = model(res_node_from, res_index_from, res_edge_from, atom2res_from, atom_node_from,
                               atom_index_from, atom_edge_from, node_pos_from, basic_attn_from, seq_from, seq_to,
                               metal_fea, devices, inputPT, foldx)

                label = label.to(devices).type(torch.float)
                out = pred_y.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                pdb_tag = ''.join(pdb_tag)
                df = df._append({'PDB': pdb_tag.split('_')[1], 'Chain_ID': pdb_tag.split('_')[5],
                                 'FromAA': pdb_tag.split('_')[4], 'ToAA': pdb_tag.split('_')[6],
                                 'Pdb_pos': pdb_tag.split('_')[3], 'Label': mean(label),
                                 'Predict': mean(out)}, ignore_index=True)
                test_predict.append(mean(out))
                test_y.append(mean(label))

        df.to_excel(outputdir_file + '{}_fold.xlsx'.format(fold_num), index=False)
        fcvtest_regression(test_predict, test_y,output_result)
        del model

if __name__ == '__main__':
    main()

