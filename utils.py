import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from scipy.stats import chisquare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

def classify_with_md(test_data, test_labels, pred,mahanalobis_dist_ts):

    """
    Perform classification with robust MD.
    """
    from sklearn.metrics import f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    anom_list= []

    # Reconstruciton loss
    for i in range(test_data.shape[0]):
        pred_i = pred[i,:]
        #print ("pred shape:", pred_i.shape)
        test_data_i = test_data[i,:]
        #print("test_data_i shape:", test_data_i.shape)
        recons_err = np.mean((test_data_i - pred_i) ** 2)
        md_dist = mahanalobis_dist_ts[i]
        anom_score = (0.05 * recons_err ) + (0.95 * md_dist)
        anom_list.append (anom_score)

    #print ("test labels:", test_labels[:50] )
    anomaly_score = dict(enumerate(anom_list))
    print ("anomaly score:", anomaly_score) #

    anom_list = [tensor.item() for tensor in anom_list]
    #print("anom list :", (anom_list))

    anom_list = np.array(anom_list)


    # This is the range of the distribution of normal data with robust MD distance
    anom_indices = [key for key, value in anomaly_score.items() if 0.2 >= value >=0.05 ] #robust MD distnace of normal data

    #Calculate best precision, recall, accuracy comparing all threholds
    lower = 0.01
    upper = 0.05

    pred_levels =  np.where((anom_list <lower) | (anom_list > upper),1,0)

    accuracy = accuracy_score(test_labels, pred_levels)
    F1 = f1_score(test_labels, pred_levels)
    precision = precision_score(test_labels, pred_levels)
    recall = recall_score(test_labels, pred_levels)

    print ("Accuracy,Precision,Recall:", accuracy,precision, recall)
    confusion_mat = confusion_matrix(test_labels, pred_levels)
    print ("CONF MAT:", confusion_mat)
    


    return accuracy, precision, recall


def mse_and_corr(test_data, test_labels, preds, targets_len):
    """
    targets and preds must have shape [time_steps, samples, variables]
    targets_len must have shape [samples,]
    """
    mse_list = []
    mae_list = []
    mse_list_norm = []
    corr_list = []
    targets_len = targets_len[0]

    for i in range(test_data.shape[0]):
            len_i = targets_len

            test_data_i = test_data[i,:len_i]
            pred_i = preds[i,:len_i]

            mse_list.append(np.mean((test_data_i - pred_i) ** 2))
            mae_list.append(np.abs(test_data_i - pred_i))
            corr_list.append(np.corrcoef(test_data_i.flatten(), pred_i.flatten())[0, 1])

    #print ("mse_list:", mse_list)
    tot_mse = np.mean(mse_list)
    #print ("TOTAL MSE ANOM:", tot_mse)

    tot_corr = np.mean(corr_list)
    tot_mae = np.mean(mae_list)

    return tot_mse, tot_corr, tot_mae
