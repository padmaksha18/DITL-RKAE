import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
from TS_datasets import getBlood
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import classify_with_knn, interp_data, mse_and_corr, dim_reduction_plot
import math
from scipy import stats
import scipy
import os
import datetime
from scipy.stats import gaussian_kde
from math import sqrt
from math import log
from torch import optim
from torch.autograd import Variable
from math import sqrt
from math import log
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import numpy as np
import scipy
from scipy.special import rel_entr
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import normalize

dim_red = 1  # perform PCA on the codes and plot the first two components
plot_on = 1  # plot the results, otherwise only textual output is returned
interp_on = 0  # interpolate data (needed if the input time series have different length)
tied_weights = 0  # train an AE where the decoder weights are the econder weights transposed
lin_dec = 0  # train an AE with linear activations in the decoder

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--code_size", default=20, help="size of the code", type=int)
parser.add_argument("--w_reg", default=0.001, help="weight of the regularization in the loss function", type=float)
parser.add_argument("--a_reg", default=0.2, help="weight of the kernel alignment", type=float)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--batch_size", default=500, help="number of samples in each batch", type=int)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--hidden_size", default=30, help="size of the code", type=int)
args = parser.parse_args()
print(args)

# ================= DATASET =================
(train_data, train_labels, train_len, _, K_tr,
 valid_data, _, valid_len, _, K_vs,
 test_data_orig, test_labels, test_len, _, K_ts) = getBlood(kernel='TCK',
                                                            inp='zero')  # data shape is [T, N, V] = [time_steps, num_elements, num_var]
# print ("test labels:", test_labels)
# sort test data (for a better visualization of the inner product of the codes)

test_data = test_data_orig
train_data = train_data
valid_data = valid_data
test_data = test_data

print(
    '\n**** Processing Blood data: Tr{}, Vs{}, Ts{} ****\n'.format(train_data.shape, valid_data.shape, test_data.shape))

input_length = train_data.shape[1]  # same for all inputs

# ================= GRAPH =================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

encoder_inputs = train_data
prior_k = K_tr

# Save entropies
entropy_list = []

# # ----- ENCODER -----

input_length = encoder_inputs.shape[1]
print("INPUT ")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.We1 = torch.nn.Parameter(
            torch.FloatTensor(input_length, args.hidden_size).uniform_(-1 / math.sqrt(input_length),
                                                                       1 / math.sqrt(input_length)))
        self.We2 = torch.nn.Parameter(
            torch.FloatTensor(args.hidden_size, args.code_size).uniform_(-1 / math.sqrt(args.hidden_size),
                                                                         1 / math.sqrt(args.hidden_size)))

        self.be1 = torch.nn.Parameter(torch.zeros([args.hidden_size]).float())
        self.be2 = torch.nn.Parameter(torch.zeros([args.code_size]).float())

        self.Wd1 = torch.nn.Parameter(
            torch.FloatTensor(args.code_size, args.hidden_size).uniform_(-1 / math.sqrt(args.code_size),
                                                                         1 / math.sqrt(args.code_size)))
        self.Wd2 = torch.nn.Parameter(
            torch.FloatTensor(args.hidden_size, input_length).uniform_(-1 / math.sqrt(args.hidden_size),
                                                                       1 / math.sqrt(args.hidden_size)))

        self.bd1 = torch.nn.Parameter(torch.zeros([args.hidden_size]).float())
        self.bd2 = torch.nn.Parameter(torch.zeros([input_length]).float())

    def encoded_codes(self, encoder_inputs):
        hidden_1 = torch.tanh(torch.matmul(encoder_inputs.float(), self.We1) + self.be1).float()
        # print ("hidden_1 shape:", hidden_1.shape)
        code = torch.tanh(torch.matmul(hidden_1, self.We2).float() + self.be2).float()
        return code

    def decoder(self, encoder_inputs):
        code = self.encoded_codes(encoder_inputs).float()

        # code = torch.sigmoid(code)
        hidden_2 = torch.tanh(torch.matmul(code, self.Wd1).float() + self.bd1)
        # activation = torch.relu(hidden_2)
        dec_out = torch.matmul(hidden_2, self.Wd2).float() + self.bd2
        # dec_out = torch.sigmoid(dec_out)
        return dec_out


def mahanalobisdist(code):
    '''
    Calculates the mahalanobis distance
    between 2 points of the data
    '''

    # print("CODE shape:", code.shape)
    code = code.detach().numpy()
    cov = np.zeros((code.shape[1], code.shape[1]))

    for i in range(code.shape[1]):
        for j in range(code.shape[1]):
            mad_i = np.median(np.abs(code[:, i] - np.median(code[:, i])))
            mad_j = np.median(np.abs(code[:, j] - np.median(code[:, j])))

            cov[i, j] = np.dot((np.divide(code[:, i] - np.median(code[:, i]), mad_i)),
                               np.divide(code[:, j] - np.median(code[:, j]), mad_j))
            cov[i, j] = cov[i, j] / code.shape[0]

    inv = np.linalg.pinv(cov)

    # Num of MD dist
    delta = np.zeros((code.shape[0], code.shape[1]))
    mdist_lst = []

    for i in range(code.shape[0]):  ## 100 * 9
        # mean = np.mean(code, axis = 0)  # 9 * 1
        median = np.median(code, axis=0)  # 9 * 1   ## feature-wise median
        delta[i, :] = code[i, :] - median
        mdist = np.abs(np.dot(np.dot(np.transpose(delta[i, :]), inv), (delta[i, :])))  # 1*9 9*9 9*1
        mdist = np.sqrt(mdist)
        mdist_lst.append(mdist)

    mahanalobis_dist = np.mean(mdist_lst)
    mahanalobis_dist = torch.from_numpy(np.asarray(mahanalobis_dist))
    # print("mdist :", mahanalobis_dist)

    return mahanalobis_dist


def mahanalobisdist_test(code_ts, code_tr):
    '''
    Calculates the mahalanobis distance
    between 2 points of the data
    '''
    # print("CODE shape:", code.shape)
    code_tr = code_tr.detach().numpy()
    code_ts = code_ts.detach().numpy()

    # create covariance matrix
    cov = np.zeros((code_tr.shape[1], code_tr.shape[1]))

    for i in range(code_tr.shape[1]):
        for j in range(code_tr.shape[1]):
            mad_i = np.median(np.abs(code_tr[:, i] - np.median(code_tr[:, i])))  ## mad of feature 1
            mad_j = np.median(np.abs(code_tr[:, j] - np.median(code_tr[:, j])))  ## mad of feature 2
            cov[i, j] = np.dot((np.divide(code_tr[:, i] - np.median(code_tr[:, i]), mad_i)),
                               np.divide(code_tr[:, j] - np.median(code_tr[:, j]), mad_j))
            cov[i, j] = cov[i, j] / code_tr.shape[0]

    inv = np.linalg.pinv(cov)

    # Num of MD dist
    delta = np.zeros((code_ts.shape[0], code_ts.shape[1]))
    mdist_lst = []

    for i in range(code_ts.shape[0]):  ## 100 * 9
        # mean = np.mean(code, axis = 0)  # 9 * 1
        median = np.median(code_tr, axis=0)  # 9 * 1   ## feature-wise median
        delta[i, :] = code_ts[i, :] - median
        mdist = np.abs(np.dot(np.dot(np.transpose(delta[i, :]), inv), (delta[i, :])))  # 1*9 9*9 9*1
        mdist = np.sqrt(mdist)
        mdist_lst.append(mdist)

    mahanalobis_dist_ts = torch.from_numpy(np.asarray(mdist_lst))
    # print("mdist :", mahanalobis_dist)
    return mahanalobis_dist_ts


def calculate_gram_mat(X, sigma):  # required only for codes
    """calculate gram matrix for variables x
        Args:
        x: random variable with two dimensional (N,d).
        sigma: kernel size of x (Gaussain kernel)
    Returns:
        Gram matrix (N,N)
    """
    x = X.view(X.shape[0], -1)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    dist = -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    return torch.exp(-dist / sigma)


def renyi_entropy(code, data, sigma):  # code is batch * latent dim
    # calculate entropy for single variables x (Eq.(9) in paper)
    #         Args:
    #         x: random variable with two dimensional (N,d).
    #         sigma: kernel size of x (Gaussain kernel)
    #         alpha:  alpha value of renyi entropy
    #     Returns:
    #         renyi alpha entropy of x.

    alpha = 2

    if data == "latent":
        # code_k = calculate_gram_mat(code, sigma)
        # sigma = kernel_smoothing(code_k, code, sigma)

        # calculate kernel with new updated sigma
        code_k = calculate_gram_mat(code, sigma)
        code_k = code_k / torch.trace(code_k)
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv, eigvec = torch.linalg.eigh(code_k)
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        # entropy = -torch.sum(eig_pow)

    elif data == "prior":  # For prior, RBF kernel is pre-computed. Just calculate entropy.
        k = code / torch.trace(code)
        # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
        eigv, eigvec = torch.linalg.eigh(k)
        eig_pow = eigv ** alpha
        entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
        # entropy = -torch.sum(eig_pow)

    return entropy


def joint_entropy(code, prior, s_x, s_y):  # x = code (batch * feats), y = prior kernel (bacth * batch)

    """calculate joint entropy for random variable x and y (Eq.(10) in paper)
        Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        alpha:  alpha value of renyi entropy
    Returns:
        joint entropy of x and y.
    """

    alpha = 2

    # s_x = kernel_smoothing(x, s_x)
    code_k = calculate_gram_mat(code, s_x)
    prior_k = prior
    # prior_k = calculate_gram_mat(prior, s_y) ## prior latent kernel 100 * 29

    k = torch.mul(code_k, prior_k)
    k = k / torch.trace(k)
    # eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eigv, eigvec = torch.linalg.eigh(k)
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    # entropy = torch.sum(eig_pow)

    return entropy


def entropy_loss(code, prior_kernel, normalize, phase, epoch):  ## calculate MI # x = code , y = prior

    """calculate Mutual information between random variables x and y
    Args:
        x: random variable with two dimensional (N,d).
        y: random variable with two dimensional (N,d).
        s_x: kernel size of x
        s_y: kernel size of y
        normalize: bool True or False, noramlize value between (0,1)
    Returns:
        Mutual information between x and y (scale)

    """
    # global s_x
    s_x = 4  # code
    s_y = 1  # prior 4,2

    # entropy of code. code is batch * latent dimension
    Hx = renyi_entropy(code, "latent", sigma=s_x)

    # entropy of prior ##For prior, RBF kernel is pre-computed. sigma is not considered
    Hy = renyi_entropy(prior_kernel, "prior", sigma=s_y)

    # joint entropy
    # Hxy = joint_entropy(x, y, s_x, s_y)
    Hxy = joint_entropy(code, prior_kernel, s_x, s_y)

    if normalize:
        # Ixy = Hx + Hy - Hxy
        Ixy = ((Hx * Hy) / (Hxy * Hxy))
        Ixy = Ixy / (torch.max(Hx, Hy))
        # Ixy = torch.log2(Ixy)

    else:
        # Ixy = Hx + Hy - Hxy
        Ixy = ((Hx * Hy) / (Hxy * Hxy))
        Ixy = Ixy / (torch.max(Hx, Hy))
        # Ixy = torch.log2(Ixy)

    return Ixy


# Initialize model
model = Model()
# model = model.double()

# trainable parameters count
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total parameters: {}'.format(total_params))

# Optimizer
# optimizer = torch.optim.Adam(model.parameters(),args.learning_rate)
optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0.001, amsgrad=False)

# ============= TENSORBOARD =============
writer = SummaryWriter()

# ================= TRAINING =================

# initialize training variables
time_tr_start = time.time()
batch_size = args.batch_size
max_batches = train_data.shape[0] // batch_size
loss_track = []
kloss_track = []
entrpy_loss_track = []
mahanalobis_dist_list = []
min_vs_loss = np.infty
model_dir = "logs/dkae_models/m_0.ckpt"

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

##############################################################################
# Training code
##############################################################################

try:
    for ep in range(args.num_epochs):

        # shuffle training data
        idx = np.random.permutation(train_data.shape[0])
        train_data_s = train_data[idx, :]
        K_tr_s = K_tr[idx, :][:, idx]

        for batch in range(max_batches):
            fdtr = {}
            fdtr["encoder_inputs"] = train_data_s[(batch) * batch_size:(batch + 1) * batch_size, :]
            fdtr["prior_K"] = K_tr_s[(batch) * batch_size:(batch + 1) * batch_size,
                              (batch) * batch_size:(batch + 1) * batch_size]

            encoder_inputs = (fdtr["encoder_inputs"].astype(float))
            encoder_inputs = torch.from_numpy(encoder_inputs)
            # print("SHAPE ENCODER_INP:", (encoder_inputs.size()))

            prior_K = (fdtr["prior_K"].astype(float))
            prior_K = torch.from_numpy(prior_K)
            # print ("SHAPE PRIOR K TRAIN:", prior_K.size())

            dec_out = model.decoder(encoder_inputs)

            code_tr = model.encoded_codes(encoder_inputs)
            code_tr = code_tr.float()
            # print ("CODE TR SHAPE:", code_tr.size())
            # print("CODE TR:", code_tr)

            reconstruct_loss = torch.mean((dec_out - encoder_inputs) ** 2)
            reconstruct_loss = reconstruct_loss.float()

            # This is just to calculate the regularization term
            encoder_in_val = torch.from_numpy(valid_data)
            code_vs = model.encoded_codes(encoder_in_val)
            dec_out_val = model.decoder(encoder_in_val)

            # This is to handle the error "numpy.linalg.LinAlgError: SVD did not converge" which arises rarely
            # even when the latent code is full rank whiel taking the inverse
            try:
                mahanalobis_dist = mahanalobisdist(code_tr)
                # mahanalobis_dist = np.mean(mdist_lst)
                # print ("MAHALANOBIS DIST TRAIN:", mahanalobis_dist)
            except Exception as e:
                print("Error in train MD:", e)

            # reconstruct_loss_reg = 1 / torch.std((dec_out_val - encoder_in_val) ** 2)
            reconstruct_loss_reg = 0.05
            # print("RECONS LOSS REG:", (reconstruct_loss_reg))

            # Mahalanobis regularizer
            # mahalanobis_reg = mahanalobisdist_reg(code_tr, code_vs)
            mahalanobis_reg = 0.95
            mahanalobis_dist = mahanalobisdist(code_tr)
            # mahalanobis_reg = torch.std(mahanalobis_dist)
            # print("MAHA LOSS REG:", (mahalanobis_reg))

            # handle the error torch._C._LinAlgError: torch.linalg.eigh: The algorithm failed to converge because
            # the input matrix is ill-conditioned or has too many repeated eigenvalues

            # try:
            entrpy_loss = entropy_loss(code_tr, prior_K, True, "train", ep)
            entrpy_loss = -entrpy_loss
            entrpy_loss = entrpy_loss.float()
            # print ("ENTRPY LOSS:", (entrpy_loss))
            # except Exception as e:
            #     print ("Error in train entropy:", e)

            # Regularization L2 loss
            reg_loss = 0

            parameters = torch.nn.utils.parameters_to_vector(model.parameters())

            for tf_var in parameters:
                reg_loss += torch.mean(torch.linalg.norm(tf_var))

            # tot_loss = reconstruct_loss + args.w_reg * reg_loss + args.a_reg * entrpy_loss
            # tot_loss = (mahalanobis_reg * mahanalobis_dist) + args.w_reg * reg_loss + args.a_reg * entrpy_loss
            tot_loss = (reconstruct_loss_reg * reconstruct_loss) + (
                        mahalanobis_reg * mahanalobis_dist) + args.w_reg * reg_loss + args.a_reg * -entrpy_loss
            tot_loss = tot_loss.float()
            # print("TOTAL LOSS:", tot_loss)

            # Backpropagation
            optimizer.zero_grad()
            # tot_loss.backward(retain_graph=True)
            # print ("TOT LOSS:", tot_loss)
            tot_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            # optimizer.apply_gradients(zip(clipped_gradients, parameters))
            optimizer.step()

            # tot_l0oss = tot_loss.detach()

            loss_track.append(tot_loss)
            # kloss_track.append(k_loss)
            entrpy_loss_track.append(entrpy_loss)

        # check training progress on the validations set (in blood data valid=train)
        if ep % 50 == 0:
            print('Ep: {}'.format(ep))
            fdvs = {}

            fdvs["encoder_inputs"] = valid_data
            fdvs["prior_K"] = K_vs

            encoder_inp = (fdvs["encoder_inputs"].astype(float))

            encoder_inp = torch.from_numpy(encoder_inp)
            # print("SHAPE ENCODER_INP:", (encoder_inp.size()))

            prior_K_vs = (fdvs["prior_K"].astype(float))

            prior_K_vs = torch.from_numpy(prior_K_vs)
            # print("SHAPE PRIOR K VAL:", prior_K_vs.size()) ## 1500 * 1500

            code_vs = model.encoded_codes(encoder_inp)
            # print("SHAPE CODE_VS VAL:", code_vs.size()) ## 1500 * 9
            # print("CODE_VS VAL:", code_vs)  ## 1500 * 9

            dec_out_val = model.decoder(encoder_inp)
            # print ("DEC OUT VAL:", dec_out_val)

            try:
                # Calculate entropy of latent code for val data
                entrpy_loss_val = entropy_loss(code_vs, prior_K_vs, True, "validation", ep)  # takes time
                entrpy_loss_val = -entrpy_loss_val
                # print("ENTRPY_LOSS_VAL:", (entrpy_loss_val))
            except Exception as e:
                print("error in VAL entropy:", e)
                break

            reconstruct_loss_val = torch.mean((dec_out_val - encoder_inp) ** 2)
            # print("reconstruct_loss_val:", reconstruct_loss_val)

            # reconstruct_loss_val_reg = 1 / torch.std(((dec_out_val - encoder_inp) ** 2))
            reconstruct_loss_val_reg = 0.05
            # print("reconstruct_loss_val_reg:", reconstruct_loss_val_reg)

            try:
                mahanalobis_dist = mahanalobisdist(code_vs)
                # mahanalobis_dist = np.mean(mahanalobis_dist)
                # print("MAHALANOBIS DIST VAL:", mahanalobis_dist)
                # mahanalobis_dist_list.append(mahanalobis_dist)
            except Exception as e:
                print("Error in VAL MD:", e)
                break

            # Mahalanobis regularizer
            # mahalanobis_reg_val = mahanalobisdist_reg(code_tr,code_vs)
            mahalanobis_reg_val = 0.95
            # mahalanobis_dist = mahanalobisdist(code_vs)
            # mahalanobis_reg_val = np.std(mahalanobis_dist)
            # print("MAHA LOSS REG:", (mahalanobis_reg_val))

            # continue

            # tot_loss_val = (0.25 * reconstruct_loss_val) + ( 0.75 * mahanalobis_dist) # take MD reg for validation as 1
            tot_loss_val = reconstruct_loss_val_reg * reconstruct_loss_val + mahalanobis_reg_val * mahanalobis_dist
            # tot_loss_val = reconstruct_loss_val

            writer.add_scalar("reconstruct_loss", reconstruct_loss_val, ep)
            writer.add_scalar("entrpy_loss", entrpy_loss_val, ep)

            # print ("loss_track    :", loss_track)
            # print ("entrpy_loss_track:", entrpy_loss_track)

            print('VS r_loss=%.8f,entropy_loss=%.8f -- TR r_loss=%.8f, entropy_loss=%.8f' % (
                tot_loss_val, entrpy_loss_val, torch.mean(torch.stack(loss_track[-50:])),
                torch.mean(torch.stack(entrpy_loss_track[-50:]))))

            # Save model yielding best results on validation
            if tot_loss_val < min_vs_loss:
                min_vs_loss = tot_loss_val
                torch.save(model, model_dir)
                torch.save(model.state_dict(), 'logs/dkae_models/best-model-parameters.pt')

            # save_path = saver.save(sess, model_name)


except KeyboardInterrupt:
    print('training interrupted')

time_tr_end = time.time()
print('Tot training time: {}'.format((time_tr_end - time_tr_start) // 60))

# # ================= TEST =================
print('************ TEST ************ \n>>restoring from:' + model_dir + '<<')

model_test = torch.load(model_dir)

encoder_inputs = torch.from_numpy(train_data)
tr_code = model_test.encoded_codes(encoder_inputs)

test_data = torch.from_numpy(test_data)
# print ("TEST DATA SHAPE:", np.shape(test_data))

ts_code = model_test.encoded_codes(test_data)
# ts_code = ts_code.detach().numpy()

# dec_out = model.encoder_decoder(encoder_inputs_te)
pred = model_test.decoder(test_data)  # decoder is doing encoding also.

pred = pred.detach().numpy()
test_data = test_data.detach().numpy()

recons_loss_test = np.mean((pred - test_data) ** 2)
print("recons_loss_test:", recons_loss_test)
# recons_loss = np.mean((pred - test_data) ** 2)

mahanalobis_dist_test = mahanalobisdist(ts_code)
# mahanalobis_dist_test = np.mean (mahanalobis_dist_test)
print("MAHALANOBIS DIST TRAIN:", mahanalobis_dist_test)

# sample wise MD
mahanalobis_dist_ts = mahanalobisdist_test(ts_code, tr_code)


tot_loss = recons_loss_test
# tot_loss = (0.05 * recons_loss_test) + (0.95 * mahanalobis_dist_test)
print('Test loss: %.5f' % (tot_loss))

# reverse transformations
# print("Test data shape:" , test_data_orig.shape)
# pred = np.reshape(pred, (test_data_orig.shape[1], test_data_orig.shape[0], test_data_orig.shape[2]))
pred = np.reshape(pred, (test_data_orig.shape[1], test_data_orig.shape[0]))
pred = np.transpose(pred, axes=[1, 0])

test_data = test_data_orig
# print("test data shape:", test_data.shape)

# MSE and corr
test_mse, test_corr, tot_mae = mse_and_corr(test_data, test_labels[:], pred, test_len)
print('Test MSE: %.5f\nTest Pearson correlation: %.3f\nTest MAE: %.5f' % (test_mse, test_corr, tot_mae))

# kNN classification on the codes
print("train labels shape:", train_labels.shape)

tr_code = tr_code.detach().numpy()
ts_code = ts_code.detach().numpy()

print("Test labels:", test_labels.shape)
acc, f1, auc = classify_with_knn(test_data, test_labels[:], pred, mahanalobis_dist_ts)
print('kNN -- acc: %.3f, F1: %.3f, AUC: %.3f' % (acc, f1, auc))

# dim reduction plots
if dim_red:
    dim_reduction_plot(ts_code, test_labels, 1)

writer.close()
