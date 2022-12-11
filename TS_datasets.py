import numpy as np
import scipy.io
from utils import ideal_kernel
import pandas as pd

"""
Data manager for loading blood data and (precomputed) TCK kernel
"""


def getBlood(kernel='TCK', inp='zero'):
    #df = pd.read_csv("df_kae_vrif_3k.csv")
    df = pd.read_csv("df_kae_cic_ids_5k_2k_norm_far.csv")
    #df = pd.read_csv("df_kae_nsl_kdd_5k_2k.csv")

    # df = df[['core:freq_lower_edge', 'core:freq_upper_edge','core:sample_start', 'core:freq_center', 'core:bandwidth', "Anomaly Flag"]]
    train_data = df.iloc[0:5000, :-1]

    # train_labels = np.asarray(blood_data['Y'])
    #train_labels = df["Anomaly Flag"].iloc[0:7000]
    train_labels = df["Label"].iloc[0:5000]
    train_data = train_data.to_numpy()
    train_labels = train_labels.to_numpy()

    # train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    train_len = [train_data.shape[0] for _ in range(train_data.shape[0])]

    # ----- test -------
    # test_data = blood_data['Xte']
    # test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    # test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]

    test_data = df.iloc[5000:6000, :-1]
    #print("Test data 2:", test_data.shape)
    #test_labels = df["Anomaly Flag"].iloc[7000:13000]
    test_labels = df["Label"].iloc[5000:6000]
    #print ("test labels data:", test_labels[:25])
    test_data = test_data.to_numpy()
    test_labels = test_labels.to_numpy()
    # test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    test_len = [test_data.shape[0] for _ in range(test_data.shape[0])]

    # valid == train
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len

    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data

    if kernel == 'TCK':
        # K_tr = blood_data['Ktrtr']
        #K_tr = np.load('K_tr_nsl_kdd_5k_2k.npy')
        K_tr = np.load('K_tr_cic_ids_5k_2k.npy')
        #K_tr = np.load('K_tr_vrif_3k.npy')
        print("K_tr:", K_tr.shape)
        K_vs = K_tr
        K_ts = np.load('K_near_te_cic_ids_5k_2k.npy')
        #K_ts = np.load('K_te_nsl_kdd_5k_2k.npy')
        #K_ts = np.load('K_te_vrif_3k.npy')
        print("K_ts:", K_ts.shape)
    # else:
    #     K_tr = ideal_kernel(train_labels)
    #     K_vs = ideal_kernel(valid_labels)
    #     K_ts = ideal_kernel(test_labels)

    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)