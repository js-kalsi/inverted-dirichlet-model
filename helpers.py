import sys
from math import pow

import numpy as np
from scipy.special import polygamma
from sklearn.preprocessing import normalize


def load_dataset(file_name):
    """

    @param file_name:
    @return:
    """
    data = np.genfromtxt(file_name, delimiter=',')
    return data[:, :-1], data[:, -1]


def split_data_by_label(labels, img_pixels):
    """

    @param labels:
    @param img_pixels:
    @return:
    """
    unique_clusters = np.unique(labels)
    clusters = {}
    for index, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(img_pixels[index])
    return clusters, unique_clusters, len(clusters[0][0])


def method_of_moment(no_of_clusters, cluster_set, dim):
    """

    @param no_of_clusters:
    @param cluster_set:
    @param dim:
    @return:
    """
    alpha = np.zeros((no_of_clusters, dim + 1))
    for label in cluster_set:
        alpha_sum = 0
        for d in range(dim):
            cluster_set[label] = np.asarray(cluster_set[label])
            mean = np.mean(cluster_set[label][:, [d]])
            den = np.var(cluster_set[label][:, [d]]) + sys.float_info.epsilon
            alpha_d_pls_one = ((pow(mean, 2) + mean) / den) + 2
            alpha_sum += alpha_d_pls_one
            alpha[label][d] = mean * (alpha_d_pls_one - 1)
        alpha[label][dim] = np.mean(alpha_sum)
    return normalize(alpha)


def posterior_estimator(pdf, mix):
    """

    @param pdf:
    @param mix:
    @return:
    """
    return np.asarray([(mix * p_v) / np.sum(mix * p_v) for p_v in pdf]).reshape(len(pdf), mix.size)


def g_estimation(data_set, size, alpha, posterior, dim):
    """

    @param data_set:
    @param size:
    @param alpha:
    @param posterior:
    @param dim:
    @return:
    """
    data_set = np.concatenate((data_set, np.full((size, 1), 1)), axis=1)
    pixel_log = np.asarray([np.log(data / np.sum(data)) for data in data_set])
    return np.asarray([g_matrix_generator(aV, posterior[:, [index]], pixel_log, dim) for index, aV in enumerate(alpha)])


def g_matrix_generator(alpha, posterior, log_pixels, dim):
    """

    @param alpha:
    @param posterior:
    @param log_pixels:
    @param dim:
    @return:
    """
    return np.sum(posterior * (
            polygamma(0, np.sum(alpha)) - polygamma(0, alpha).reshape(1, dim + 1) + log_pixels.reshape(len(log_pixels),
                                                                                                       dim + 1)),
                  axis=0).reshape(dim + 1, 1)


def hessian(dim, posterior, alpha):
    """
    
    @param dim:
    @param posterior:
    @param alpha:
    @return:
    """
    h_diagonal = []
    h_constant = []
    h_a = []
    for index, a_vector in enumerate(alpha):
        p_sum = np.sum(posterior[:, index])
        a_tri_gamma = polygamma(1, a_vector)
        a_tri_gamma_sum = polygamma(1, np.sum(a_vector))
        h_diagonal.append(np.diag(1 / (-1 * p_sum * a_tri_gamma)))
        h_constant.append(((a_tri_gamma_sum * np.sum(1 / a_tri_gamma)) - 1) * a_tri_gamma_sum * p_sum)
        h_a.append(((-1 / p_sum) * (1 / a_tri_gamma)).reshape(1, dim + 1))

    return np.asarray(h_diagonal), h_constant, np.asarray(h_a)


def hessian_inverse(no_of_clusters, diagonal, constant, a):
    """
    
    @param no_of_clusters: 
    @param diagonal: 
    @param constant: 
    @param a: 
    @return: 
    """
    return np.asarray([diagonal[k] + (constant[k] * np.matmul(a[k].T, a[k])) for k in range(no_of_clusters)])


def alpha_updater(alpha_set, h_inv, g, no_of_clusters, dim):
    """
    
    @param alpha_set: 
    @param h_inv: 
    @param g: 
    @param no_of_clusters: 
    @param dim: 
    @return: 
    """
    return np.asarray([alpha_set[j].reshape(dim + 1, 1) - 1.90909 * np.matmul(h_inv[j], g[j])
                       for j in range(no_of_clusters)]).reshape(no_of_clusters, dim + 1)


def cluster_drop_test(mix, alpha, cluster_drop_val, no_of_clusters, dim):
    """
    
    @param mix: 
    @param alpha: 
    @param cluster_drop_val: 
    @param no_of_clusters: 
    @param dim: 
    @return: 
    """
    mix_arr = []
    alpha_arr = []
    mix = mix[0]
    for j in range(no_of_clusters):
        if mix[j] > cluster_drop_val:
            mix_arr.append(mix[j])
            alpha_arr.append(alpha[j])
        else:
            print("Cluster having  alpha :", alpha[j], " & Mix :", j, " & Value :", mix[j], " is removed!")

    mix_arr = np.asarray(mix_arr)
    alpha_arr = np.asarray(alpha_arr)
    return mix_arr.reshape(1, mix_arr.size), alpha_arr.reshape(len(alpha_arr), dim + 1), mix_arr.size


def convergence_test(alpha, convergence_val):
    """
    
    @param alpha: 
    @param convergence_val: 
    @return: 
    """
    alpha_size = len(alpha)
    if alpha_size < 2:
        return False
    if alpha[alpha_size - 1].shape == alpha[alpha_size - 2].shape:
        return np.all(convergence_val >= (alpha[alpha_size - 1] - alpha[alpha_size - 2]))
    print("Two or more value of Alpha Exist but their shape differs !")
    return False


def cluster_density_evaluation(labels):
    """
    
    @param labels: 
    @return: 
    """
    zero_arr = []
    one_arr = []
    for j in labels:
        if j == 0:
            zero_arr.append(j)
        else:
            one_arr.append(j)
    return len(zero_arr), len(one_arr)
