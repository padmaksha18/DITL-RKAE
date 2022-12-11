import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from scipy.stats import chisquare


def dim_reduction_plot(data, label, block_flag):
    """
    Compute linear PCA and scatter the first two components
    """

    PCA_model = TruncatedSVD(n_components=3).fit(data)
    data_PCA = PCA_model.transform(data)
    idxc1 = np.where(label == 0)
    idxc2 = np.where(label == 1)

    plt.scatter(data_PCA[idxc1, 0], data_PCA[idxc1, 1], s=80, c='y', marker='o', linewidths=0, label='healthy')
    plt.scatter(data_PCA[idxc2, 0], data_PCA[idxc2, 1], s=80, c='r', marker='^', linewidths=0, label='infected')


    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.title('PCA of the codes')
    plt.legend(scatterpoints=1, loc='best')
    # plt.show(block=block_flag)
    plt.savefig(str(block_flag) + '.jpg')

    # from sklearn.preprocessing import StandardScaler
    # from sklearn.pipeline import Pipeline
    # pca = PCA()
    # pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    # Xt = pipe.fit_transform(data)
    # plot = plt.scatter(Xt[:, 0], Xt[:, 1], c=label)
    # plt.legend(handles=plot.legend_elements()[0], labels=list(label))
    # plt.savefig(str(block_flag) + '.jpg')


def ideal_kernel(labels):
    """
    Compute the ideal kernel K
    An entry k_ij = 0 if i and j have different class
    k_ij = 1 if i and j have same class
    """
    K = np.zeros([labels.shape[0], labels.shape[0]])

    for i in range(labels.shape[0]):
        k = labels[i] == labels
        k.astype(int)
        K[:, i] = k[:, 0]
    return K


def interp_data(X, X_len, restore=False, interp_kind='linear'):
    """
    Interpolate data to match the same maximum length in X_len
    If restore is True, data are interpolated back to their original length
    data are assumed to be time-major
    interp_kind: can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
    """

    [T, N, V] = X.shape
    X_new = np.zeros_like(X)

    # restore original lengths
    if restore:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=T)
            t_new = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            for v in range(V):
                x_n_v = X[:, n, v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:X_len[n], n, v] = f(t_new)

    # interpolate all data to length T
    else:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            t_new = np.linspace(start=0, stop=X_len[n], num=T)
            for v in range(V):
                x_n_v = X[:X_len[n], n, v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:, n, v] = f(t_new)

    return X_new



def classify_with_knn(test_data, test_labels, pred,mahanalobis_dist_ts):

    """
    Perform classification with knn.
    """
    #from sklearn.neighbors import KNeighborsClassifier
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
    print ("anomaly score:", anomaly_score) # >>> {0: 'a', 1: 'b', 2: 'c'}

    anom_list = [tensor.item() for tensor in anom_list]
    print("anom list :", (anom_list))

    anom_list = np.asarray(anom_list)
    # Find normal indices for the best distribution
    # normal_indices_1 = [key for key, value in anomaly_score.items() if 0.05 >= value >=0.008 ] #CIC-IDS Normal 0.05 >= value >=0.008
    # print ("normal_indices:", normal_indices_1)
    # print ("normal samples:", len(normal_indices_1))

    ##Anomaly data
    #anom_indices_1 = [key for key, value in anomaly_score.items() if 0.009 > value] #CIC-IDS Anomaly 0.009 >= value  0.05 <= value <=0.009
    # print ("anom_indices_1:", anom_indices_1)
    # print ("anom samples:", len(anom_indices_1))

    #lowers = [0.009,0.01,0.02,0.03,0.04,0.05]
    uppers = np.arange(0.05,0.4,0.01)

    for upper in uppers:
    #for lower in lowers:

            #pred_levels = np.where(np.asarray(anom_list)<= lower , 1,0) #or np.where(np.asarray(anom_list).any() >= 0.07 , 1,0 )
            pred_levels = np.where(np.asarray(anom_list) >= upper, 1,0)

            accuracy = accuracy_score(test_labels, pred_levels)
            F1 = f1_score(test_labels, pred_levels)
            precision = precision_score(test_labels, pred_levels)
            #print ("precision:", precision)
            AUC = roc_auc_score(test_labels, pred_levels)
            recall = recall_score(test_labels, pred_levels)

            print ("Upper:", upper)
            print ("Accuracy,Precision,Recall, AUC:", accuracy,precision, recall, AUC)
            confusion_mat = confusion_matrix(test_labels, pred_levels)
            print ("CONF MAT:", confusion_mat)
            tn, fp, fn, tp = confusion_matrix(test_labels, pred_levels).ravel()


    return accuracy, F1, AUC



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
    # print ("targets_len:", targets_len)
    # print("targets shape:", targets.shape)
    # print("preds shape:", preds.shape)
    #print ("test_labels:", test_labels.shape)

    for i in range(test_data.shape[0]):
            len_i = targets_len
            #print ("len_i:", len_i)
            test_data_i = test_data[i,:len_i]
            #print("test_data_i:", test_data_i)

            # pred_i = preds[:len_i,i,:]
            pred_i = preds[i,:len_i]
            #print("pred_i:", pred_i)

            mse_list.append(np.mean((test_data_i - pred_i) ** 2))
            mae_list.append(np.abs(test_data_i - pred_i))
            corr_list.append(np.corrcoef(test_data_i.flatten(), pred_i.flatten())[0, 1])

    #print ("mse_list:", mse_list)
    tot_mse = np.mean(mse_list)
    #print ("TOTAL MSE ANOM:", tot_mse)

    tot_corr = np.mean(corr_list)
    tot_mae = np.mean(mae_list)

    return tot_mse, tot_corr, tot_mae


  # plotting labelled histogram
    # plt.hist(anomaly_score_lst, 2000, range=[0.1, 5], facecolor='gray', align='mid')
    # plt.xlabel('scores')
    # plt.ylabel('count')
    # plt.savefig('histo.jpg')