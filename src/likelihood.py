import numpy as np
from numba import float64,float32,jit,boolean,vectorize,int64
from numba.types import UniTuple, Tuple,void
import math



quick_pi = -1*np.log(np.sqrt(2*np.pi))
sqrt_2 = np.sqrt(2)


@vectorize([float64(float64)],nopython=True)
def sgn_smooth(x):
    return 1.0/(1+math.exp(-10.0*x + 5.0 ))


@vectorize([float64(float64)],nopython=True)
def sgn(x):
    if x>0.5: return 1.0
    else: return 0.0

@vectorize([float64(boolean)],nopython=True)
def asfloat(x):
    if x: return 1.0
    else: return 0.0

@jit(float64(float64),nopython=True)
def phi_1(val):
    return quick_pi-0.5*np.square(val)


@jit(float64(float64),nopython=True)
def phi_2(val):
    return .5*(1.0+math.erf(val/sqrt_2))


@jit(float64(float64,float64,float64),nopython=True)
def trunc_norm_logpdf(val,mean,std):
    if val> 1.0 or val<0.0:
        return -10000000.000
    else:
        term_a = phi_1((val-mean)/std)
        term_b = phi_2((1.0-mean)/std)
        term_c = phi_2(-1.0*mean/std)
        return term_a/(std*(term_b-term_c))





@jit(float64(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64))
def full_log_likelihood_single(i_cells:np.ndarray,j_genes:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float,  y_scale: float):

    z = sgn(np.dot(i_cells,j_genes.T))
    inv_z = 1.0-z
    #compute the loglikelihood of the expression data
    p_x = np.sum(np.multiply(z,prob_matrix_a) + np.multiply(inv_z,prob_matrix_b))



    z_total = np.sum(z ,0) * (1.0/(i_cells.shape[0]))
    p_y = sum([trunc_norm_logpdf(z_total[index],y[index],sigma_y) for index in range(y.size)])

    return p_x+y_scale*p_y



@jit(boolean(float64[:,:],float64[:,:],int64,int64,float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def opt_jk_numba_single(i_cells:np.ndarray,j_genes:np.ndarray,j:int,k:int,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float,  y_scale: float, z_arr:np.ndarray, z_count: np.ndarray):


    denom = i_cells.shape[0]
    val = j_genes[j,k]
    z = sgn(z_count[:, j])
    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[:, j])) + \
                            np.sum(np.multiply(1.0 - z, prob_matrix_b[:, j]))
    if val == 0.0:
        #case of changing 0 to 1

        changed_z_count = i_cells[:, k] + z_count[:, j]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 0.0, changed_z_count == 1.0))
        changed_z_val = z_arr[j]+num_changed

    else:
        #case of changing 1 to 0

        changed_z_count = z_count[:, j] - i_cells[:, k]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 1.0, changed_z_count == 0.0))
        changed_z_val = z_arr[j] - num_changed

    z_changed = sgn(changed_z_count)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[:, j])) + \
                              np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[:, j]))
    y_stay_log_likelihood = trunc_norm_logpdf(z_arr[j]/denom, y[j], sigma_y)
    y_change_log_likelihood = trunc_norm_logpdf(changed_z_val/denom, y[j], sigma_y)
    stay_log_likelihood = x_stay_log_likelihood+(y_scale*y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood+(y_scale*y_change_log_likelihood)

    if change_log_likelihood-stay_log_likelihood > .000001:
        j_genes[j,k] = 1.0 - j_genes[j,k]
        z_arr[j] = changed_z_val
        z_count[:, j] = changed_z_count
        return True
    else:
        return False


@jit(float64[:](float64[:,:],float64[:,:],int64,int64,float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def opt_ik_numba_single(i_cells: np.ndarray, j_genes: np.ndarray, i: int, k: int,
           prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
           sigma_y: float, y_scale: float, z_arr: np.ndarray, z_count: np.ndarray):


    val = i_cells[i,k]
    z = sgn(z_count[i,:])
    denom = i_cells.shape[0]
    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z, prob_matrix_b[i,:]))
    if val == 0.0:

        changed_z_count = z_count[i,:]+j_genes[:,k]
        #nums_changed = asfloat(changed_z_count == 1.0)
        nums_changed = asfloat(np.logical_and(z_count[i,:] == 0.0, changed_z_count == 1.0))
        changed_z_vals=  z_arr  + nums_changed



    else:
        changed_z_count = z_count[i, :] - j_genes[:, k]
        #nums_changed = asfloat(changed_z_count == 0.0)
        nums_changed = asfloat(np.logical_and(z_count[i, :] == 1.0, changed_z_count == 0.0))
        changed_z_vals = z_arr - nums_changed

    z_changed = sgn(changed_z_count)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[i,:]))
    y_stay_log_likelihood = 0.0

    y_change_log_likelihood = 0.0
    for g in range(y.size):
            y_stay_log_likelihood += trunc_norm_logpdf(z_arr[g]/denom, y[g], sigma_y)
            y_change_log_likelihood += trunc_norm_logpdf(changed_z_vals[g]/denom, y[g], sigma_y)


    stay_log_likelihood = x_stay_log_likelihood + (y_scale * y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood + (y_scale * y_change_log_likelihood)


    if change_log_likelihood - stay_log_likelihood > .0001:
        i_cells[i, k] = 1.0 - i_cells[i, k]
        z_count[i,:] = changed_z_count
        #return True
        return changed_z_vals

    else:
        #return False
        return z_arr



@jit(void(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def greedy_optimize_j_numba_single(i_cells: np.ndarray, j_genes: np.ndarray,
                      prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
                      sigma_y: float, y_scale: float,z_arr:np.ndarray, z_count:np.ndarray):
    for j in range(j_genes.shape[0]):
        for k in range(j_genes.shape[1]):
           opt_jk_numba_single(i_cells,j_genes,j,k,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count)



@jit(void(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def greedy_optimize_i_numba_single(i_cells:np.ndarray,j_genes:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, y_scale: float, z_arr:np.ndarray, z_count:np.ndarray):
    for i in range(i_cells.shape[0]):
        for k in range(i_cells.shape[1]):
            z_arr =  opt_ik_numba_single(i_cells,j_genes,i,k,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count)











