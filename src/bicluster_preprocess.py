import numpy as np
from typing import List,Tuple
from scipy.stats import norm,kurtosis,skew
from skimage.filters import threshold_otsu
from sklearn import mixture
from fixed_mixture import GaussianMixture



def value_to_bimodal_prob(data:np.ndarray, bimodal_params:List[Tuple[float,float,float,float]]) -> \
        Tuple[np.ndarray,np.ndarray]:
    '''

    :param data: Cell x Gene numpy array
    :param bimodal_params: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    :return: prob_matrix_a : matrix log probability of data for the first mode,
    prob_matrix_b matrix log probability of data for the second mode
    '''
    prob_matrix_a = np.zeros(data.shape)
    prob_matrix_b = np.zeros(data.shape)
    vnorm = np.vectorize(norm.logpdf)
    for i in range(data.shape[1]):
        prob_matrix_a[:, i] = vnorm(data[:, i],bimodal_params[i][0],bimodal_params[i][1])
        prob_matrix_b[:, i] = vnorm(data[:, i], bimodal_params[i][2], bimodal_params[i][3])
    return prob_matrix_a,prob_matrix_b


def value_to_bimodal_prob_extended(data:np.ndarray, bimodal_params:List[Tuple[float,float,float,float]]) -> \
        Tuple[np.ndarray,np.ndarray]:
    '''

    :param data: Cell x Gene numpy array
    :param bimodal_params: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    :return: prob_matrix_a : matrix log probability of data for the first mode,
    prob_matrix_b matrix log probability of data for the second mode
    '''
    prob_matrix_a = np.zeros(data.shape)
    prob_matrix_b = np.zeros(data.shape)
    prob_matrix_c = np.zeros(data.shape)
    prob_matrix_d = np.zeros(data.shape)
    lvnorm = np.vectorize(norm.logpdf)
    vnorm = np.vectorize(norm.pdf)
    for i in range(data.shape[1]):
        p1 = vnorm(data[:, i],bimodal_params[i][0],bimodal_params[i][1])
        p2 = vnorm(data[:, i],bimodal_params[i][2],bimodal_params[i][3])
        p_total = p1+p2
        prob_matrix_c[:,i] = np.log(np.divide(p1,p_total))
        prob_matrix_d[:, i] = np.log(np.divide(p2,p_total))
        #mean_diff = bimodal_params[i][0] - bimodal_params[i][2]
        prob_matrix_a[:, i] = lvnorm(data[:, i], bimodal_params[i][0], bimodal_params[i][1])
        prob_matrix_b[:, i] = lvnorm(data[:, i], bimodal_params[i][2], bimodal_params[i][3])
        #prob_matrix_a[:, i] = lvnorm(data[:, i]-bimodal_params[i][2],0,bimodal_params[i][3]+np.square(mean_diff))
        #prob_matrix_b[:, i] = lvnorm(data[:, i]-bimodal_params[i][2], 0, bimodal_params[i][3])
        #prob_matrix_a[:, i] = vnorm(bimodal_params[i][0], data[:, i],bimodal_params[i][1])
        #prob_matrix_b[:, i] = vnorm(bimodal_params[i][2], data[:, i],  bimodal_params[i][3])
        #prob_matrix_a[:, i] = vnorm(data[:, i], 3, 1)
        #prob_matrix_b[:, i] = vnorm(data[:, i], 0, 1)
    #print(prob_matrix_a[np.isinf(np.log(prob_matrix_a))])
    return prob_matrix_a,prob_matrix_b, prob_matrix_c, prob_matrix_d


def find_bimodal_params2(data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''
    bimodal_params = []
    mix = mixture.BayesianGaussianMixture(n_components=2, covariance_type="spherical", n_init=100)
    for i in range(data.shape[1]):
        gene_vec = data[:,i]
        #fit a unimodal guassian
        mean_b, std_b = norm.fit(gene_vec)
        var_b = np.sqrt(std_b)


        col_vec = gene_vec.reshape(gene_vec.size, -1)

        #fit a bimodal guassian and take the second peak
        mix.fit(col_vec)
        if mix.means_[1][0] > mix.means_[0][0]:
            mean_a = mix.means_[1][0]
            var_a = np.sqrt(mix.covariances_[1])
        else:
            mean_a = mix.means_[0][0]
            var_a = np.sqrt(mix.covariances_[0])

        bimodal_params.append((mean_a,var_a,mean_b,var_b))
        print(bimodal_params[i])
    return bimodal_params

def find_bimodal_params1(data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''
    bimodal_params = []
    mix = mixture.GaussianMixture(n_components=2, covariance_type="spherical", max_iter=200)
    for i in range(data.shape[1]):
        gene_vec = data[:,i]
        #fit a unimodal guassian


        col_vec = gene_vec.reshape(gene_vec.size, -1)

        #fit a bimodal guassian and take the second peak
        mix.fit(col_vec)
        if mix.means_[1][0] > mix.means_[0][0]:
            mean_a = mix.means_[1][0]
            var_a = np.sqrt(mix.covariances_[1])
            #var_a = np.square(mix.covariances_[1])

            mean_b = mix.means_[0][0]
            var_b = np.sqrt(mix.covariances_[0])
            #var_b = np.square(mix.covariances_[0])
        else:
            mean_a = mix.means_[0][0]
            var_a = np.sqrt(mix.covariances_[0])
            #var_a = np.square(mix.covariances_[0])

            mean_b = mix.means_[1][0]
            var_b = np.sqrt(mix.covariances_[1])
            #var_b = np.square(mix.covariances_[1])
        bimodal_params.append((mean_a,var_a,mean_b,var_b))
    return bimodal_params



def find_bimodal_params(data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''
    bimodal_params = []
    mix = mixture.GaussianMixture(n_components=2, covariance_type="spherical")
    for i in range(data.shape[1]):
        gene_vec = data[:,i]
        sarle_score = sarle(gene_vec)
        if sarle_score < 5.0/10.0:
            mean_b, std_b = norm.fit(gene_vec)
            var_b = np.sqrt(std_b)
            mean_a = np.max(gene_vec)*2
            var_a = .5
            bimodal_params.append((mean_a, var_a, mean_b, var_b))
        else:
            bimodal_params.append(otsu_bimod(gene_vec))
    return bimodal_params

def sarle(arr:np.ndarray):
    numerator = skew(arr) +1
    denom = kurtosis(arr) + 3*(arr.size-1)*(arr.size-1)/((arr.size-2)*(arr.size)-3)
    return numerator/denom



def find_bimodal_paramsx(data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''
    bimodal_params = []
    mix = mixture.BayesianGaussianMixture(n_components=2, covariance_type="spherical", n_init=100)
    for i in range(data.shape[1]):
        gene_vec = data[:,i]

        #fit a unimodal guassian
        #mean_b, std_b = norm.fit(gene_vec)
        #var_b = np.sqrt(std_b)
        #mean_a, var_a, _,_ = otsu_bimod(gene_vec)

        bimodal_params.append(otsu_bimod(gene_vec))
    return bimodal_params


def find_bimodal_params_otsu(data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''
    bimodal_params = np.zeros(shape=(data.shape[1],4))
    for i in range(data.shape[1]):
        gene_vec = data[:,i]
        mean_a, std_a, mean_b, std_b =otsu_bimod(gene_vec)
        bimodal_params[i,:] = np.asarray([mean_a,std_a,mean_b,std_b])
    return bimodal_params



def otsu_bimod(values:np.ndarray)->Tuple[float,float]:
    otsu_threshold =threshold_otsu(values)

    upper_peak_values = values[values>=otsu_threshold]
    lower_peak_values = values[values<otsu_threshold]
    mean_a, std_a = norm.fit(upper_peak_values)
    #mean_b, std_b = norm.fit( lower_peak_values )
    mean_b, std_b = norm.fit(values)



    return mean_a, std_a, mean_b, std_b


def all_cols_otsu(mat:np.ndarray)-> np.ndarray:
    return np.apply_along_axis(otsu_bimod_l2,0,mat)


def otsu_bimod_l2(values:np.ndarray)->float:
    otsu_threshold =threshold_otsu(values)
    return float(np.sum(values>=otsu_threshold))/ float(values.size)


def simple_learn_bimodal_paramas(data:np.ndarray, y_data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :param y_data Dataset x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''

    bimodal_params = []
    for i in range(data.shape[1]):
        gene_vec = data[:, i]
        if y_data[i] <0.05:
            y_data[i] = 0.05
        if y_data[i] > 0.95:
            y_data[i] = 0.95
        split_index = int(round((1.0-y_data[i])*gene_vec.size))
        sorted_gene_vec = np.sort(gene_vec)
        mean_a, std_a = norm.fit(sorted_gene_vec[split_index:])
        mean_b, std_b = norm.fit(sorted_gene_vec[:split_index])
        params_tuple = (mean_a,std_a,mean_b,std_b)
        bimodal_params.append(params_tuple)
    return bimodal_params



def fixed_learn_bimodal_paramas(data:np.ndarray, y_data:np.ndarray)->List[Tuple[float,float,float,float]]:
    '''

    :param data: Cell x Gene numpy array
    :param y_data Dataset x Gene numpy array
    :return: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    '''

    bimodal_params = []
    for i in range(data.shape[1]):
        gene_vec = data[:, i]
        if y_data[i]<.05: y_data[i] = .05
        means = np.matrix([np.amax(gene_vec),np.amin(gene_vec)]).T

        mix = GaussianMixture(n_components=2, covariance_type="spherical",weights_init=[y_data[i],1-y_data[i]],
                                      means_init=means)
        mix.fit(np.matrix(gene_vec).T)
        if mix.means_[1][0] > mix.means_[0][0]:
            mean_a = mix.means_[1][0]
            var_a = np.sqrt(mix.covariances_[1])
            mean_b = mix.means_[0][0]
            var_b = np.sqrt(mix.covariances_[0])
        else:
            mean_a = mix.means_[0][0]
            var_a = np.sqrt(mix.covariances_[0])
            mean_b = mix.means_[1][0]
            var_b = np.sqrt(mix.covariances_[1])

        params_tuple = (mean_a,var_a,mean_b,var_b)
        bimodal_params.append(params_tuple)
    return bimodal_params


def norm_chip_seq(exp_data:np.ndarray,epi_data:np.ndarray):
    epi_data_predict_norm = np.zeros(epi_data.shape)
    for d in range(epi_data.shape[1]):
        pure_peaks = epi_data[:,d] == np.sum(epi_data,1)
        pure_exp = exp_data[:,pure_peaks]
        pi_list = []
        for g in range(pure_exp.shape[1]):
            values = pure_exp[:,g]
            sarle_score = sarle(values)
            if sarle_score >= 5.0 / 9.0:
                otsu_threshold = threshold_otsu(values)
                pi = (values[values >= otsu_threshold]).size/ values.size
                pi_list.append(pi)
            else:
                pi_list.append(np.nan)
        pi_arr = np.asarray(pi_list)
        if np.sum(epi_data[pure_peaks,d]) != 0.0:
            coef = np.nanmean(epi_data[pure_peaks,d]/pi_arr)
            epi_data_predict_norm[:,d] = epi_data[:,d] /coef
        else:
            epi_data_predict_norm[:, d] = epi_data[:,d]*0.0
    return epi_data_predict_norm















