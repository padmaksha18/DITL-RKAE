import numpy as np
import scipy.io
import pandas as pd


def getBlood(kernel='TCK', inp='zero'):
    df = pd.read_csv(
        "/content/gdrive/MyDrive/OOD_generalization/mate-mi-reg-model/DATA_2/DDOS_GOLDEN_EYE.csv")
        
    df.replace([np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    #df.loc[df['Label'] == 2.0, 'Label'] = 1.0

    train_data = df.iloc[20000:40000, :-1]

    train_labels = df['Label'].iloc[20000:40000]

    train_data = train_data.to_numpy()
    train_labels = train_labels.to_numpy()

    # train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    train_len = [train_data.shape[0] for _ in range(train_data.shape[0])]

    # ----- test -------
    test_data = df[20000:40000]#.loc[df['Label'] == 1]
    
    test_labels = test_data['Label']
    # test_data = test_normal.iloc[ 15000:80000, :-1]
    test_data = test_data.iloc[:, :-1]
    print("TEST DATA SHAPE:", test_data.shape)
    
    test_data = test_data.to_numpy()
    test_labels = test_labels.to_numpy()
    # test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    test_len = [test_data.shape[0] for _ in range(test_data.shape[0])]

    # valid == train
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len

    # # target outputs
    train_data = train_data
    valid_targets = valid_labels
    test_targets = test_labels

    return (train_data, train_labels, train_len, train_labels,
            valid_data, valid_labels, valid_len, valid_targets,
            test_data, test_labels, test_len, test_targets)
