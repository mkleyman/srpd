import numpy as np
from scipy.stats import norm
from skimage.filters import threshold_otsu
from SimulatedData import create_data
from typing import List,Tuple
import numba
from numba import float64,float32,jit,boolean,vectorize
from numba.types import UniTuple


@jit(float64(float64,float64,float64),nopython=True)
def normal_pdf(x:float,mu:float,sigma:float)->float:
    if sigma == 0.0 : sigma = .001
    part_a = 1.0/np.sqrt(2.0*np.pi*np.square(sigma))
    part_b = np.exp(-1.0*np.square(x-mu)/(2.0*np.square(sigma)))
    return part_a*part_b






@jit(UniTuple(float64[:],2)(float64[:],float64,float64,float64,float64,float64,float64),nopython=True)
def e_step_numba(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    z_0_prob = np.zeros(x_vals.shape)
    z_1_prob = np.zeros(x_vals.shape)
    for i in range(x_vals.shape[0]):
        a = pi_0*normal_pdf(x_vals[i],mu_0,sigma_0)
        b = pi_1 * normal_pdf(x_vals[i], mu_1, sigma_1)
        if a == 0.0: a = b*0.001
        if b == 0.0: b = a*0.001
        z_0_prob[i] = a/(a+b)
        z_1_prob[i] = b / (a + b)
    return z_0_prob,z_1_prob

@jit(UniTuple(float64,6)(float64[:],float64[:],float64[:]),nopython=True)
def m_step_numba(x_vals,z_0_prob,z_1_prob):
    n_0 = np.sum(z_0_prob)
    n_1 = np.sum(z_1_prob)

    if n_0 <= 0.0: n_0 = 0.001
    if n_1 <= 0.0: n_1 = 0.001


    if n_0 == 0.0:
        mu_0 = np.amin(x_vals)
        sigma_0 = 0.01
    else:
        mu_0 = np.sum(np.multiply(x_vals,z_0_prob))/n_0
        sigma_0 = np.sqrt(np.sum(np.multiply(np.square(x_vals - mu_0), z_0_prob)) / n_0)
    if n_1 == 0.0:
        mu_1 = np.amax(x_vals)
        sigma_1 = 0.01
    else:
        mu_1 = np.sum(np.multiply(x_vals,z_1_prob))/n_1
        sigma_1 = np.sqrt(np.sum(np.multiply( np.square(x_vals-mu_1),z_1_prob))/n_1)

    pi_0 = n_0/x_vals.size
    pi_1 = n_1/x_vals.size

    return pi_0,pi_1,mu_0,mu_1,sigma_0,sigma_1

@jit(boolean(float64,float64,float64),nopython=True)
def isclose(a,b,error):
    return error> np.abs(a-b)

@jit(UniTuple(float64,4)(float64[:],float64,float64,float64,float64,float64,float64),nopython=True)
def em_algorithm_fixed_pi_numba(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    mu_0_new, mu_1_new, sigma_0_new, sigma_1_new =  1.0, 1.0, 1.0, 1.0
    iteration = 0
    while not (isclose(mu_0, mu_0_new, .01) and isclose(mu_1, mu_1_new, .01) and
                   isclose(sigma_0, sigma_0_new, .01) and isclose(sigma_1, sigma_1_new, .01) or iteration==500):
        if iteration>0:
            mu_0, mu_1, sigma_0, sigma_1 = mu_0_new, mu_1_new, sigma_0_new, sigma_1_new
        iteration+=1
        z_0_prob, z_1_prob = e_step_numba(x_vals, pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1)
        _,_,mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = m_step_numba(x_vals,z_0_prob,z_1_prob)
    return mu_0_new, mu_1_new, sigma_0_new, sigma_1_new

@jit(UniTuple(float64,6)(float64[:],float64,float64,float64,float64,float64,float64),nopython=True)
def em_algorithm_numba(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = 1.0,1.0,1.0,1.0,1.0,1.0
    iteration = 0
    while not(isclose(pi_0,pi_0_new,.01) and isclose(pi_1,pi_1_new,.01) and  isclose(mu_0,mu_0_new,.01)\
            and isclose(mu_1,mu_1_new,.01) and isclose(sigma_0,sigma_0_new,.01) and isclose(sigma_1,sigma_1_new,.01)):
        if iteration>0:
            pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1 = pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new

        iteration+=1
        z_0_prob, z_1_prob = e_step_numba(x_vals, pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1)
        pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = m_step_numba(x_vals,z_0_prob,z_1_prob)
    return pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new

def e_step(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    z_0_prob = np.zeros(x_vals.shape)
    z_1_prob = np.zeros(x_vals.shape)
    for i in range(x_vals.shape[0]):
        a = pi_0*norm.pdf(x_vals[i],mu_0,sigma_0)
        b = pi_1 * norm.pdf(x_vals[i], mu_1, sigma_1)
        if a == 0.0: a = b*0.001
        if b == 0.0: b = a*0.001
        z_0_prob[i] = a/(a+b)
        z_1_prob[i] = b / (a + b)
    return z_0_prob,z_1_prob

def m_step(x_vals,z_0_prob,z_1_prob):
    n_0 = np.sum(z_0_prob)
    n_1 = np.sum(z_1_prob)



    if n_0 == 0.0:
        mu_0 = np.amin(x_vals)
        sigma_0 = 0.01
    else:
        mu_0 = np.sum(np.multiply(x_vals,z_0_prob))/n_0
        sigma_0 = np.sqrt(np.sum(np.multiply(np.square(x_vals - mu_0), z_0_prob)) / n_0)
    if n_1 == 0.0:
        mu_1 = np.amax(x_vals)
        sigma_1 = 0.01
    else:
        mu_1 = np.sum(np.multiply(x_vals,z_1_prob))/n_1
        sigma_1 = np.sqrt(np.sum(np.multiply( np.square(x_vals-mu_1),z_1_prob))/n_1)

    pi_0 = n_0/x_vals.size
    pi_1 = n_1/x_vals.size

    return pi_0,pi_1,mu_0,mu_1,sigma_0,sigma_1

def em_algorithm(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = 1.0,1.0,1.0,1.0,1.0,1.0
    iteration = 0
    while not(np.isclose(pi_0,pi_0_new,.1) and np.isclose(pi_1,pi_1_new,.1) and  np.isclose(mu_0,mu_0_new,.1)\
            and np.isclose(mu_1,mu_1_new,.1) and np.isclose(sigma_0,sigma_0_new,.1) and np.isclose(sigma_1,sigma_1_new,.1)):
        if iteration>0:
            pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1 = pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new

        iteration+=1
        z_0_prob, z_1_prob = e_step(x_vals, pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1)
        pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = m_step(x_vals,z_0_prob,z_1_prob)
    return pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new

def em_algorithm_fixed_pi(x_vals,pi_0,pi_1,mu_0,mu_1, sigma_0, sigma_1):
    pi_0_new, pi_1_new, mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    iteration = 0
    while not (np.isclose(mu_0, mu_0_new, .01) and np.isclose(mu_1, mu_1_new, .01) and
                   np.isclose(sigma_0, sigma_0_new, .01) and np.isclose(sigma_1, sigma_1_new, .01) or iteration==500):
        if iteration>0:
            mu_0, mu_1, sigma_0, sigma_1 = mu_0_new, mu_1_new, sigma_0_new, sigma_1_new
        iteration+=1
        z_0_prob, z_1_prob = e_step(x_vals, pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1)
        _,_,mu_0_new, mu_1_new, sigma_0_new, sigma_1_new = m_step(x_vals,z_0_prob,z_1_prob)
    return mu_0_new, mu_1_new, sigma_0_new, sigma_1_new


def otsu_init(values:np.ndarray):
    otsu_threshold =threshold_otsu(values)

    upper_peak_values = values[values>=otsu_threshold]
    lower_peak_values = values[values<otsu_threshold]
    mean_a, std_a = norm.fit(upper_peak_values)
    mean_b, std_b = norm.fit(lower_peak_values)
    pi_a =  upper_peak_values.size / values.size
    pi_b = lower_peak_values.size/ values.size

    return pi_a,pi_b,mean_a, mean_b, np.sqrt(std_a) , np.sqrt(std_b)


def simple_init(values:np.ndarray):
    mean,var = norm.fit(values)
    mean_a,mean_b = mean+var,mean-var
    #std_a,std_b = np.sqrt(var),np.sqrt(var)


    return mean_a, mean_b, var, var

@jit(UniTuple(float64,4)(float64[:]),nopython=True)
def simple_init_numba(values:np.ndarray):
    output = np.zeros(4,dtype=float64)
    #mean = np.mean(values)
    mx = np.amax(values)
    mn = np.amin(values)
    std = np.std(values)
    #output[0] = mean+output[2]
    #output[1] = mean - output[2]
    return mx,mn,std,std

@jit(UniTuple(float64,6)(float64[:]),nopython=True)
def simple_init_numba_naive(values:np.ndarray):
    output = np.zeros(4,dtype=float64)
    mean = np.mean(values)
    std = np.std(values)
    #output[0] = mean+output[2]
    #output[1] = mean - output[2]
    return .5,.5,mean+std,mean-std,std,std

def make_bimodal_params(sample_data:np.ndarray,epi_data:np.ndarray)->List[Tuple[float,float,float,float]]:
    param_list = []
    for i in range(sample_data.shape[1]):
        row = sample_data[:,i]
        mu_0, mu_1, sigma_0, sigma_1 = simple_init(row)

        mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_fixed_pi(row, epi_data[i], 1.0-epi_data[i], mu_0, mu_1, sigma_0, sigma_1)
        #mean, std = norm.fit(values)
        params = (mu_0,sigma_0,mu_1,sigma_1)
        param_list.append(params)
    return param_list

@jit(float64[:,:](float64[:,:],float64[:]),nopython=True)
def make_bimodal_params_numba(sample_data:np.ndarray,epi_data:np.ndarray)->np.ndarray:
    #param_arr = np.zeros((sample_data.shape[1],4), dtype=float64)
    param_arr = np.zeros((sample_data.shape[1], 4))
    for i in range(sample_data.shape[1]):
        row = sample_data[:,i]
        mu_0_init, mu_1_init, sigma_0_init, sigma_1_init = simple_init_numba(row)
        #mu_0, mu_1, sigma_0, sigma_1 = 0,1,.5,.5
        mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_fixed_pi_numba(row, epi_data[i], 1.0-epi_data[i], mu_0_init, mu_1_init,
                                                                   sigma_0_init, sigma_1_init)
        if not( np.isfinite(mu_0) and np.isfinite(mu_1) and np.isfinite(sigma_0) and np.isfinite(sigma_1)):
            param_arr[i, 0], param_arr[i, 1], param_arr[i, 2], param_arr[i, 3] = mu_0_init, mu_1_init, sigma_0_init, sigma_1_init
        elif mu_0>mu_1 :
            param_arr[i,0],  param_arr[i,1],param_arr[i,2],param_arr[i,3]= mu_0,sigma_0,mu_1,sigma_1
        else:
            param_arr[i, 2], param_arr[i, 3], param_arr[i, 0], param_arr[i, 1] = mu_0, sigma_0, mu_1, sigma_1
        #param_arr[i,4] = epi_data[i]
    return param_arr

'''
@jit(float64[:,:](float64[:,:],float64[:], boolean[:]),nopython=True)
def make_bimodal_params_numba_bicurve(sample_data:np.ndarray,epi_data:np.ndarray, gene_curve:np.ndarray)->np.ndarray:
    param_arr = np.zeros((sample_data.shape[1],4), dtype=float64)
    for i in range(sample_data.shape[1]):
        row = sample_data[:,i]
        mu_0, mu_1, sigma_0, sigma_1 = simple_init_numba(row)
        #mu_0, mu_1, sigma_0, sigma_1 = 0,1,.5,.5
        if gene_curve[i]:
            epi_val = epi_data[i]
        else:
            epi_val = 1.0 - epi_data[i]
        mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_fixed_pi_numba(row, epi_val, 1.0-epi_val, mu_0, mu_1, sigma_0, sigma_1)
        if mu_0>mu_1 == gene_curve  :
            param_arr[i,0],  param_arr[i,1],param_arr[i,2],param_arr[i,3]= mu_0,sigma_0,mu_1,sigma_1
        else:
            param_arr[i, 2], param_arr[i, 3], param_arr[i, 0], param_arr[i, 1] = mu_0, sigma_0, mu_1, sigma_1
    return param_arr'''


@jit(float64[:,:](float64[:,:]),nopython=True)
def make_bimodal_params_naive_numba(sample_data:np.ndarray)->np.ndarray:
    param_arr = np.zeros((sample_data.shape[1],4), dtype=float64)
    for i in range(sample_data.shape[1]):
        row = sample_data[:,i]
        mu_0, mu_1, sigma_0, sigma_1 = simple_init_numba(row)
        pi_0_new, pi_1_new, mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_numba(row, .5,.5, mu_0, mu_1, sigma_0, sigma_1)
        if mu_0>mu_1:
            param_arr[i,0],  param_arr[i,1],param_arr[i,2],param_arr[i,3]= mu_0,sigma_0,mu_1,sigma_1
        else:
            param_arr[i, 2], param_arr[i, 3], param_arr[i, 0], param_arr[i, 1] = mu_0, sigma_0, mu_1, sigma_1
        #param_arr[i,4] = epi_data[i]
    return param_arr

@vectorize([float64(float64,float64,float64)],nopython=True)
def normal_logpdf(x,mu,sigma):
    part_a = 1.0/np.sqrt(2.0*np.pi*np.square(sigma))
    part_b = -1.0*np.square(x-mu)/(2.0*np.square(sigma))
    return np.log(part_a)+part_b

@jit(float64[:,:](float64[:,:]),nopython=True)
def make_bimodal_params_numba_naive(sample_data:np.ndarray)->np.ndarray:
    param_arr = np.zeros((sample_data.shape[1],4), dtype=float64)
    for i in range(sample_data.shape[1]):
        row = sample_data[:,i]
        pi_0,pi_1,mu_0, mu_1, sigma_0, sigma_1 = simple_init_numba_naive(row)
        pi_0, pi_1,mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_numba(row, pi_0,pi_1, mu_0, mu_1, sigma_0, sigma_1)
        if mu_0>mu_1:
            param_arr[i,0],  param_arr[i,1],param_arr[i,2],param_arr[i,3]= mu_0,sigma_0,mu_1,sigma_1
        else:
            param_arr[i, 2], param_arr[i, 3], param_arr[i, 0], param_arr[i, 1] = mu_0, sigma_0, mu_1, sigma_1
    return param_arr


@jit(float64[:](float64[:],float64,float64),nopython=True)
def blah(x,mu,sigma):
    part_a = 1.0/np.sqrt(2.0*np.pi*np.square(sigma))
    part_b = -1.0*np.square(x-mu)/(2.0*np.square(sigma))
    return np.log(part_a)+part_b

@jit(UniTuple(float64[:,:],2)(float64[:,:],float64[:,:]),nopython=True)
def value_to_bimodal_prob_numba(data:np.ndarray, bimodal_params:np.ndarray) -> \
        Tuple[np.ndarray,np.ndarray]:
    '''

    :param data: Cell x Gene numpy array
    :param bimodal_params: List of bimodal guassian param tuples, tuple structure: mean_1, var_1, mean_2, var_2
    :return: prob_matrix_a : matrix log probability of data for the first mode,
    prob_matrix_b matrix log probability of data for the second mode
    '''
    prob_matrix_a = np.zeros(data.shape,dtype=float64)
    prob_matrix_b = np.zeros(data.shape,dtype=float64)
    for i in range(data.shape[1]):
        prob_matrix_a[:, i] = blah(data[:, i],bimodal_params[i][0],bimodal_params[i][1])
        prob_matrix_b[:, i] = blah(data[:, i], bimodal_params[i][2], bimodal_params[i][3])
    return prob_matrix_a,prob_matrix_b





if __name__ == "__main__":
    bicluster_list, sample_data, epi_data = create_data(100, 50, 5, 10, 0.0)
    for i in range(sample_data.shape[0]):
        row = sample_data[i]
        pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1 = otsu_init(row)
        #pi_0, pi_1, mu_0, mu_1, sigma_0, sigma_1 = epi_data[i], 1.0-epi_data[i], np.amax(row),np.amin(row), np.std(row)/2, np.std(row)/2
        #print("pi_0: " + str(pi_0) + " pi_1: " + str(pi_1) + " mu_0: " + str(mu_0) + " mu_1: " + str(mu_1) +
         #     " sigma_0: " + str(sigma_1) + " sigma_1: " + str(sigma_1))
        mu_0, mu_1, sigma_0, sigma_1 = em_algorithm_fixed_pi(row, epi_data[i], 1.0-epi_data[i], mu_0, mu_1, sigma_0, sigma_1)
        #print("pi_0: "+str(pi_0)+" pi_1: "+str(pi_1)+" mu_0: "+str(mu_0)+" mu_1: "+str(mu_1)+
             # " sigma_0: "+str(sigma_1)+" sigma_1: "+str(sigma_1))
        print("mu_0: "+str(mu_0)+" mu_1: "+str(mu_1)+" sigma_0: "+str(sigma_1)+" sigma_1: "+str(sigma_1)+" pi_0: " +
              str(epi_data[i]))











