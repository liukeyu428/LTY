import os
import sys
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader
import torch as th
from tools.utils import set_seed, set_save_path, Logger, save, EarlyStopping
from tools.run_tools import train_one_epoch, evaluate_one_epoch, Kl_loss, train1_one_epoch, evaluate1_one_epoch, \
     evaluate2_one_epoch, evaluate3_one_epoch, evaluate, train1, evaluate_one_epoch1, \
    train_one_epoch1, evaluate_best_epoch, evaluate_other, train_one_epochNOENV, evaluate_one_epochNOENV
from models.EEGNet import  CEDL, ATCNet, WMB_ATCNet1, \
    WMB_ATCNet_NOENV

from data.bciciv2a_process import load_bciciv2a_data_single_subject
from data.high_gamma_process import load_highgamma_data_single_subject
from data.bciciv2b_process import load_bciciv2b_data_single_subject
from data.sampler import BalanceIDSampler
from data.eegdataset import EEGDataset, EEGDataLoader
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



# 初始化设备
def train(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, 'information.txt'))
    start_epoch = 0
    # ------------------------------------------------device setting--------------------------------------------------
    device = 'cuda:0' if th.cuda.is_available() else 'cpu'

    # ------------------------------------------------data setting----------------------------------------------------
    if "high_gamma" in args.data_path:
        args.sub_num = 14
        args.class_num = 4
        load_data = load_highgamma_data_single_subject
    elif "BCICIV_2a" in args.data_path:
        args.sub_num = 9
        args.class_num = 4
        load_data = load_bciciv2a_data_single_subject
    elif "BCICIV_2b" in args.data_path:
        args.sub_num = 9
        args.class_num = 2
        load_data = load_bciciv2b_data_single_subject
    else:
        raise ValueError("only support hipgh_gamma or BCICIV_2a dataset.")
    id_list = [i + 1 for i in range(args.sub_num)] #i=1-9
    source_id_list = []
    source_X_list, source_y_list = [], []
    target_X, target_y, target_test_X, target_test_y = [None] * 4
    for i in id_list:
        if i != args.target_id:
            train_X, train_y, _, _ = load_data(args.data_path, subject_id=i, to_tensor=False)
            source_id_list.append(i)
            source_X_list.append(train_X)
            source_y_list.append(train_y)
        else:
            target_X, target_y, target_test_X, target_test_y = load_data(args.data_path, subject_id=i, to_tensor=False)
    args.source_id_list = source_id_list
    list1=[2,3,4,5,6,7,8,9]
    args.batch_size = (len(list1)+1)*64
    data_sampler = BalanceIDSampler(source_X_list, target_X, source_y_list, target_y, args.batch_size)  # 1：8加载batch
    args.batch_size = data_sampler.revised_batch_size
    data_num = data_sampler.__len__()
    train_data = EEGDataset(source_X_list, target_X, source_y_list, target_y, data_num)  # 改
    trainLoader = DataLoader(train_data, args.batch_size, shuffle=False, sampler=data_sampler, num_workers=8,
                             drop_last=False)
    test_data = EEGDataset(source_X_list=None, target_X=target_test_X, source_y_list=None, target_y=target_test_y)
    testLoader = DataLoader(test_data, args.batch_size // (len(source_id_list) + 1),
                            shuffle=False, num_workers=4, drop_last=False)
    model = CEDL(source_X_list[0].shape[-2], source_X_list[0].shape[-1], args.class_num)
    model_path = '/home/liutaoyuan/WMB_EEGNet-master/2A.OUT/无分支/12_16_17_59'
    checkpoint = th.load(os.path.join(model_path, 'model_best.pth.tar'), map_location=device)
    model.load_state_dict(checkpoint['model_classifier'])
    model.eval()
    cls_criterion = Kl_loss()
    epoch = 0
    start_time = time.time()
    avg_acc, avg_loss, all_labels, all_preds = evaluate_other(testLoader, model, device, cls_criterion, start_time,
                                               epoch, args)
    print(f"Restored Model Accuracy: {avg_acc * 100:.2f}%")
    print(f"Kappa: {cohen_kappa_score(all_labels, all_preds):.4f}")

    # 绘制归一化混淆矩阵（可选）
    class_names = ['Left hand', 'Right hand', 'Foot', 'Tongue']
    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (Target ID: {args.target_id})", fontsize=20)
    plt.xlabel("Predicted Label",fontsize=16)
    plt.ylabel("True Label",fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"new_matrix_{args.target_id}.png")
    plt.close()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"可训练参数数量: {num_params:,}")

    def estimate_model_memory(model):
        total_params = sum(p.numel() for p in model.parameters())
        total_memory = total_params * 4 / (1024 ** 2)  # 假设 float32，每个参数占 4 bytes
        return total_memory

    mem_mb = estimate_model_memory(model)
    print(f"模型内存占用（估算）: {mem_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str,
                        default='/home/liutaoyuan/WMB_EEGNet-master/BCICIV_2a/output',
                        help='Data path.')
    parser.add_argument('-f1', type=int, default=16,
                        help='the number of filters in EEGNet.')
    parser.add_argument('-target_id', type=int, default=2, help='Target id.')
    parser.add_argument('-dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('-pool', type=str, default='mean', choices=['max', 'mean'])
    parser.add_argument('-epochs', type=int, default=800, help='Number of epochs to train.')
    parser.add_argument('-lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('-adjust_lr', type=int, default=1, choices=[0, 1, 2], help='Learning rate changes over epoch.')
    parser.add_argument('-w_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-batch_size', default=576, type=int,
                        help='batch size for training')
    parser.add_argument('-early_stop', action='store_false', help='Train early stop.')
    parser.add_argument('-print_freq', type=int, default=3, help='The frequency to show training information.')
    parser.add_argument('-father_path', type=str, default='save',
                        help='The father path of models parameters, log files.')
    parser.add_argument('-seed', type=int, default='111', help='Random seed.')
    args_ = parser.parse_args()
    train(args_)
