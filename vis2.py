import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from mne.channels.layout import _auto_topomap_coords

from braindecode.visualization.plot import ax_scalp
from data.bciciv2b_process import load_bciciv2b_data_single_subject
import torch
import torch as th
import os
import sys
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader

from tools.utils import set_seed, set_save_path, Logger, save, EarlyStopping
from tools.run_tools import train_one_epoch, evaluate_one_epoch, Kl_loss, train1_one_epoch, evaluate1_one_epoch, \
     evaluate2_one_epoch, evaluate3_one_epoch, evaluate, train1, evaluate_one_epoch1, \
    train_one_epoch1, evaluate_best_epoch, evaluate_other, train_one_epochNOENV, evaluate_one_epochNOENV
from models.EEGNet import  WMB_ATCNet1, CEDLCross

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
import numpy as np
import mne
import matplotlib.pyplot as plt
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
            train_X, train_y, test_source_x, test_source_y= load_data(args.data_path, subject_id=i, to_tensor=False)
            source_id_list.append(i)
            source_X_list.append(train_X)
            source_y_list.append(train_y)
        else:
            target_X, target_y, target_test_X, target_test_y = load_data(args.data_path, subject_id=i, to_tensor=False)
    args.source_id_list = source_id_list
    data = target_X[:, :, 250:850]
    label = target_y
    # get the data and label
    # data - (samples, channels, trials)
    # label -  (label, 1)

    #data = np.transpose(data, (2, 1, 0))
    label = np.squeeze(np.transpose(label))
    '''idx = np.where(label == 3)
    if label[idx[0][0]] == 0:
        labelin = '左手'
    elif label[idx[0][0]] == 1:
        labelin = '右手'
    elif label[idx[0][0]] == 2:
        labelin = 'feet'
    elif label[idx[0][0]] == 3:
        labelin = 'tongue'
'''
    data_draw = data#[idx]

    mean_trial = np.mean(data_draw, axis=0)  # mean trial
    # use standardization or normalization to adjust
    mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)

    mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left

    # Draw topography
    biosemi_montage = mne.channels.make_standard_montage('biosemi64')  # set a montage, see mne document
    index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56,
             29]  # correspond channel
    biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
    biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')  # sample rate

    evoked1 = mne.EvokedArray(mean_trial, info)
    evoked1.set_montage(biosemi_montage)
    plt.figure(figsize=(6, 6))  # 单位是英寸，越大图越大
    #plt.figure(1)
    #im, cn = mne.viz.plot_topomap(np.mean(mean_trial, axis=1), evoked1.info, show=False)
    fig, ax = plt.subplots(figsize=(6, 6))  # 控制整体图像尺寸（单位：英寸）
    im, cn = mne.viz.plot_topomap(
        mean_ch,
        evoked1.info,
        axes=ax,
        show=False,
        outlines='head',  # 使用标准脑轮廓
        contours=6,
        sensors='k.',  # 'k.' 表示黑色圆点
    )
    #im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False)



    # ✅ 更细的轮廓线条
    for coll in ax.collections:
        try:
            coll.set_linewidth(0.5)
        except Exception:
            pass

    plt.colorbar(im)
    plt.savefig(f'./topo1/{args.target_id}.png',dpi=1000)
    print('the end')

    '''plt.figure(figsize=(6,6), constrained_layout=True)  # 正方形画布，自动布局
    im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False)
    plt.colorbar(im, shrink=0.7)
    plt.axis('off')  # 可选：隐藏坐标轴
    plt.savefig('./topo/test5.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()'''


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