import numpy as np
from typing import Callable
from scipy.stats import norm
from numba import float64,float32,jit,boolean,vectorize,int64
from numba.types import UniTuple, Tuple,void


@vectorize([float64(float64)],nopython=True)
def sgn(x):
    if x>0.5: return 1.0
    else: return 0.0

@vectorize([float64(boolean)],nopython=True)
def asfloat(x):
    if x: return 1.0
    else: return 0.0

@jit(float64(float64,float64,float64),nopython=True)
def normal_logpdf(x:float,mu:float,sigma:float)->float:
    part_a = 1.0/np.sqrt(2.0*np.pi*np.square(sigma))
    part_b = -1.0*np.square(x-mu)/(2.0*np.square(sigma))
    return np.log(part_a)+part_b



def full_log_likelihood(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, sigma_eps:float, y_scale: float):

    #z is the binary matrix N x G of of whether sample n is in the same bicluster as gene g
    z = (np.dot(i_cells,j_genes.T)>.5).astype(float)
    inv_z = 1.0-z
    #compute the loglikelihood of the expression data
    p_x = np.sum(np.multiply(z,prob_matrix_a) + np.multiply(inv_z,prob_matrix_b))

    #compute the log likelihood of epislon
    #vnorm = np.vectorize(norm.logpdf)
    p_eps = norm.logpdf(np.sum(np.square(eps)),0,sigma_eps)
    #p_eps = np.sum(vnorm(eps,0,sigma_eps))

    i_eps = np.zeros(i_cells.shape)
    for k in range(i_cells.shape[1]):
        i_eps[:,k] = eps[k]*i_cells[:,k]
    z_eps = np.dot(i_eps,j_genes.T)
    z_total = np.sum(z + z_eps,0) * (1/(i_cells.shape[0]+np.sum(i_eps)))
    p_y = sum([norm.logpdf(z_total[index],y[index],sigma_y) for index in range(y.size)])

    return -1*(p_x+y_scale*(p_eps*j_genes.shape[0]+p_y))


def full_log_likelihood_single(i_cells:np.ndarray,j_genes:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float,  y_scale: float):

    #z is the binary matrix N x G of of whether sample n is in the same bicluster as gene g
    z = (np.dot(i_cells,j_genes.T)>.5).astype(float)
    inv_z = 1.0-z
    #compute the loglikelihood of the expression data
    p_x = np.sum(np.multiply(z,prob_matrix_a) + np.multiply(inv_z,prob_matrix_b))

    #compute the log likelihood of epislon
    #vnorm = np.vectorize(norm.logpdf)


    z_total = np.sum(z ,0) * (1.0/(i_cells.shape[0]))
    p_y = sum([norm.logpdf(z_total[index],y[index],sigma_y) for index in range(y.size)])

    return p_x+y_scale*p_y


def eps_likelihood(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,y: np.ndarray,
                   sigma_y:float, sigma_eps:float):

    #z is the binary matrix N x G of of whether sample n is in the same bicluster as gene g
    z = (np.dot(i_cells,j_genes.T)>.5).astype(float)
    #compute the log likelihood of epislon
    vnorm = np.vectorize(norm.logpdf)
    p_eps = np.sum(vnorm(eps,0,sigma_eps))
    i_eps = np.zeros(i_cells.shape)
    for k in range(i_cells.shape[1]):
        i_eps[:,k] = eps[k]*i_cells[:,k]
    z_eps = np.dot(i_eps,j_genes.T)
    z_total = np.sum(z + z_eps,0) * (1/(i_cells.shape[0]+np.sum(i_eps)))
    p_y = sum([norm.logpdf(z_total[index],y[index],sigma_y) for index in range(y.size)])
    return -1*((p_eps+p_y))

def greedy_optimize_i_test(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, sigma_eps:float, y_scale: float, z_arr:np.ndarray, z_count:np.ndarray, denom:float):
    best_likelihood = full_log_likelihood(i_cells,j_genes,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,sigma_eps,y_scale)
    for i in range(i_cells.shape[0]):
        for k in range(i_cells.shape[1]):

            i_cells[i, k] = 1.0 - i_cells[i, k]

            current_likelihood = full_log_likelihood(i_cells,j_genes,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,sigma_eps,y_scale)

            i_cells[i, k] = 1.0 - i_cells[i, k]

            result =  opt_ik(i_cells,j_genes,i,k,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count,denom)


            if type(result) is tuple:
                denom = result[2]
                z_arr = result[1]
                if current_likelihood>=best_likelihood:
                    print("false positive")
                best_likelihood = current_likelihood
            elif current_likelihood<best_likelihood:
                print("false negative")

            '''
            if current_likelihood < best_likelihood:
                best_likelihood = current_likelihood
            else:
                i_cells[i, k] = 1.0 - i_cells[i, k]'''

def greedy_optimize_i(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, sigma_eps:float, y_scale: float, z_arr:np.ndarray, z_count:np.ndarray, denom:float):
    for i in range(i_cells.shape[0]):
        for k in range(i_cells.shape[1]):
            result =  opt_ik(i_cells,j_genes,i,k,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count,denom)
            if type(result) is tuple:
                denom = result[2]
                z_arr = result[1]



def greedy_optimize_j(i_cells: np.ndarray, j_genes: np.ndarray, eps: np.ndarray,
                      prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
                      sigma_y: float, sigma_eps: float, y_scale: float,z_arr:np.ndarray, z_count:np.ndarray, denom:float):
    for j in range(j_genes.shape[0]):
        for k in range(j_genes.shape[1]):
           opt_jk(i_cells,j_genes,j,k,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count,denom)




def greedy_optimize_mat(indicator_matrix:np.ndarray,func:Callable):
    best_likelihood = func(indicator_matrix)
    for i in range(indicator_matrix.shape[0]):
        for k in range(indicator_matrix.shape[1]):
            indicator_matrix[i,k] = 1.0 - indicator_matrix[i,k]
            current_likelihood = func(indicator_matrix)
            if current_likelihood < best_likelihood:
                best_likelihood = current_likelihood
            else:
                indicator_matrix[i, k] = 1.0 - indicator_matrix[i, k]
    return best_likelihood


def opt_jk(i_cells:np.ndarray,j_genes:np.ndarray,j:int,k:int,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float,  y_scale: float, z_arr:np.ndarray, z_count: np.ndarray, denom: float):

    val = j_genes[j,k]
    z = z_count[:, j].astype(bool).astype(float)
    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[:, j])) + \
                            np.sum(np.multiply(1.0 - z, prob_matrix_b[:, j]))
    if val == 0.0:
        #case of changing 0 to 1

        changed_z_count = i_cells[:, k] + z_count[:, j]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 0.0, changed_z_count == 1.0))
        changed_z_val = np.sum(eps[k]*i_cells[:,k])+z_arr[j]+num_changed

    else:
        #case of changing 1 to 0

        changed_z_count = z_count[:, j] - i_cells[:, k]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 1.0, changed_z_count == 0.0))
        changed_z_val = z_arr[j] - (np.sum(eps[k] * i_cells[:, k])+ num_changed )

    z_changed = changed_z_count.astype(bool).astype(float)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[:, j])) + \
                              np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[:, j]))
    y_stay_log_likelihood = norm.logpdf(z_arr[j]/denom, y[j], sigma_y)
    y_change_log_likelihood = norm.logpdf(changed_z_val/denom, y[j], sigma_y)
    stay_log_likelihood = x_stay_log_likelihood+(y_scale*y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood+(y_scale*y_change_log_likelihood)

    if change_log_likelihood-stay_log_likelihood > .001:
        j_genes[j,k] = 1.0 - j_genes[j,k]
        z_arr[j] = changed_z_val
        z_count[:, j] = changed_z_count
        return True
    else:
        return False


def opt_ik(i_cells: np.ndarray, j_genes: np.ndarray, i: int, k: int, eps: np.ndarray,
           prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
           sigma_y: float, y_scale: float, z_arr: np.ndarray, z_count: np.ndarray, denom:float):
    val = i_cells[i,k]
    z = z_count[i,:].astype(bool).astype(float)

    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z, prob_matrix_b[i,:]))
    if val == 0.0:

        changed_z_count = z_count[i,:]+j_genes[:,k]
        nums_changed = np.logical_and(z_count[i,:] == 0.0, changed_z_count == 1.0).astype(float)
        changed_z_vals=  z_arr + (eps[k]*j_genes[:,k]) + nums_changed
        denom_changed = denom+eps[k]


    else:
        changed_z_count = z_count[i, :] - j_genes[:, k]
        nums_changed = np.logical_and(z_count[i, :] == 1.0, changed_z_count == 0.0).astype(float)
        changed_z_vals = z_arr - (eps[k] * j_genes[:, k]) - nums_changed
        denom_changed = denom - eps[k]

    z_changed = changed_z_count.astype(bool).astype(float)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[i,:]))
    y_stay_log_likelihood = 0.0

    y_change_log_likelihood = 0.0
    for g in range(y.size):
            y_stay_log_likelihood += norm.logpdf(z_arr[g]/denom, y[g], sigma_y)
            y_change_log_likelihood += norm.logpdf(changed_z_vals[g]/denom_changed, y[g], sigma_y)


    stay_log_likelihood = x_stay_log_likelihood + (y_scale * y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood + (y_scale * y_change_log_likelihood)


    if change_log_likelihood - stay_log_likelihood > .0001:
        i_cells[i, k] = 1.0 - i_cells[i, k]
        z_count[i,:] = changed_z_count
        return (True,changed_z_vals,denom_changed,change_log_likelihood-stay_log_likelihood)

    else:
        return False

def y_log_likelihood(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,y: np.ndarray,
                   sigma_y:float, sigma_eps:float, z:np.ndarray):

    #z is the binary matrix N x G of of whether sample n is in the same bicluster as gene g
    #compute the loglikelihood of the expression data

    #compute the log likelihood of epislon
    vnorm = np.vectorize(norm.logpdf)
    p_eps = np.sum(vnorm(eps,0,sigma_eps))

    i_eps = np.zeros(i_cells.shape)
    for k in range(i_cells.shape[1]):
        i_eps[:,k] = eps[k]*i_cells[:,k]
    z_eps = np.dot(i_eps,j_genes.T)
    z_total = np.sum(z + z_eps,0) * (1/(i_cells.shape[0]+np.sum(i_eps)))
    p_y = sum([norm.logpdf(z_total[index],y[index],sigma_y) for index in range(y.size)])

    return -1*(p_eps+p_y)



@jit(boolean(float64[:,:],float64[:,:],int64,int64,float64[:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:],float64),nopython=True)
def opt_jk_numba(i_cells:np.ndarray,j_genes:np.ndarray,j:int,k:int,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float,  y_scale: float, z_arr:np.ndarray, z_count: np.ndarray, denom: float):


    val = j_genes[j,k]
    z = sgn(z_count[:, j])
    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[:, j])) + \
                            np.sum(np.multiply(1.0 - z, prob_matrix_b[:, j]))
    if val == 0.0:
        #case of changing 0 to 1

        changed_z_count = i_cells[:, k] + z_count[:, j]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 0.0, changed_z_count == 1.0))
        changed_z_val = np.sum(eps[k]*i_cells[:,k])+z_arr[j]+num_changed

    else:
        #case of changing 1 to 0

        changed_z_count = z_count[:, j] - i_cells[:, k]
        num_changed =  np.sum(np.logical_and(z_count[:, j] == 1.0, changed_z_count == 0.0))
        changed_z_val = z_arr[j] - (np.sum(eps[k] * i_cells[:, k])+ num_changed )

    z_changed = sgn(changed_z_count)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[:, j])) + \
                              np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[:, j]))
    y_stay_log_likelihood = normal_logpdf(z_arr[j]/denom, y[j], sigma_y)
    y_change_log_likelihood = normal_logpdf(changed_z_val/denom, y[j], sigma_y)
    stay_log_likelihood = x_stay_log_likelihood+(y_scale*y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood+(y_scale*y_change_log_likelihood)

    if change_log_likelihood-stay_log_likelihood > .001:
        j_genes[j,k] = 1.0 - j_genes[j,k]
        z_arr[j] = changed_z_val
        z_count[:, j] = changed_z_count
        return True
    else:
        return False


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
    y_stay_log_likelihood = normal_logpdf(z_arr[j]/denom, y[j], sigma_y)
    y_change_log_likelihood = normal_logpdf(changed_z_val/denom, y[j], sigma_y)
    stay_log_likelihood = x_stay_log_likelihood+(y_scale*y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood+(y_scale*y_change_log_likelihood)

    if change_log_likelihood-stay_log_likelihood > .001:
        j_genes[j,k] = 1.0 - j_genes[j,k]
        z_arr[j] = changed_z_val
        z_count[:, j] = changed_z_count
        return True
    else:
        return False

@jit(Tuple((boolean,float64[:],float64))(float64[:,:],float64[:,:],int64,int64,float64[:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:],float64),nopython=True)
def opt_ik_numba(i_cells: np.ndarray, j_genes: np.ndarray, i: int, k: int, eps: np.ndarray,
           prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
           sigma_y: float, y_scale: float, z_arr: np.ndarray, z_count: np.ndarray, denom:float):
    val = i_cells[i,k]
    z = sgn(z_count[i,:])

    x_stay_log_likelihood = np.sum(np.multiply(z, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z, prob_matrix_b[i,:]))
    if val == 0.0:

        changed_z_count = z_count[i,:]+j_genes[:,k]
        nums_changed = asfloat(np.logical_and(z_count[i,:] == 0.0, changed_z_count == 1.0))
        changed_z_vals=  z_arr + (eps[k]*j_genes[:,k]) + nums_changed
        denom_changed = denom+eps[k]


    else:
        changed_z_count = z_count[i, :] - j_genes[:, k]
        nums_changed = asfloat(np.logical_and(z_count[i, :] == 1.0, changed_z_count == 0.0))
        changed_z_vals = z_arr - (eps[k] * j_genes[:, k]) - nums_changed
        denom_changed = denom - eps[k]

    z_changed = sgn(changed_z_count)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[i,:]))
    y_stay_log_likelihood = 0.0

    y_change_log_likelihood = 0.0
    for g in range(y.size):
            y_stay_log_likelihood += normal_logpdf(z_arr[g]/denom, y[g], sigma_y)
            y_change_log_likelihood += normal_logpdf(changed_z_vals[g]/denom_changed, y[g], sigma_y)


    stay_log_likelihood = x_stay_log_likelihood + (y_scale * y_stay_log_likelihood)
    change_log_likelihood = x_change_log_likelihood + (y_scale * y_change_log_likelihood)


    if change_log_likelihood - stay_log_likelihood > .0001:
        i_cells[i, k] = 1.0 - i_cells[i, k]
        z_count[i,:] = changed_z_count
        #return True
        return (True,changed_z_vals,denom_changed)

    else:
        #return False
        return (False,z_arr,denom)


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
        nums_changed = asfloat(changed_z_count == 1.0)
        #nums_changed = asfloat(np.logical_and(z_count[i,:] == 0.0, changed_z_count == 1.0))
        changed_z_vals=  z_arr  + nums_changed



    else:
        changed_z_count = z_count[i, :] - j_genes[:, k]
        nums_changed = asfloat(changed_z_count == 0.0)
        #nums_changed = asfloat(np.logical_and(z_count[i, :] == 1.0, changed_z_count == 0.0))
        changed_z_vals = z_arr - nums_changed

    z_changed = sgn(changed_z_count)
    x_change_log_likelihood = np.sum(np.multiply(z_changed, prob_matrix_a[i,:])) + np.sum(np.multiply(1.0 - z_changed, prob_matrix_b[i,:]))
    y_stay_log_likelihood = 0.0

    y_change_log_likelihood = 0.0
    for g in range(y.size):
            y_stay_log_likelihood += normal_logpdf(z_arr[g]/denom, y[g], sigma_y)
            y_change_log_likelihood += normal_logpdf(changed_z_vals[g]/denom, y[g], sigma_y)


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

@jit(void(float64[:,:],float64[:,:],float64[:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:],float64),nopython=True)
def greedy_optimize_j_numba(i_cells: np.ndarray, j_genes: np.ndarray, eps: np.ndarray,
                      prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
                      sigma_y: float, y_scale: float,z_arr:np.ndarray, z_count:np.ndarray, denom:float):
    for j in range(j_genes.shape[0]):
        for k in range(j_genes.shape[1]):
           opt_jk_numba(i_cells,j_genes,j,k,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count,denom)

@jit(void(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def greedy_optimize_j_numba_single(i_cells: np.ndarray, j_genes: np.ndarray,
                      prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray, y: np.ndarray,
                      sigma_y: float, y_scale: float,z_arr:np.ndarray, z_count:np.ndarray):
    for j in range(j_genes.shape[0]):
        for k in range(j_genes.shape[1]):
           opt_jk_numba_single(i_cells,j_genes,j,k,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count)

@jit(void(float64[:,:],float64[:,:],float64[:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:],float64),nopython=True)
def greedy_optimize_i_numba(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, y_scale: float, z_arr:np.ndarray, z_count:np.ndarray, denom:float):
    for i in range(i_cells.shape[0]):
        for k in range(i_cells.shape[1]):
            result =  opt_ik_numba(i_cells,j_genes,i,k,eps,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count,denom)
            if result[0]:
                denom = result[2]
                z_arr = result[1]

@jit(void(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64,float64,float64[:],float64[:,:]),nopython=True)
def greedy_optimize_i_numba_single(i_cells:np.ndarray,j_genes:np.ndarray,
                   prob_matrix_a:np.ndarray, prob_matrix_b:np.ndarray, y: np.ndarray,
                   sigma_y:float, y_scale: float, z_arr:np.ndarray, z_count:np.ndarray):
    for i in range(i_cells.shape[0]):
        for k in range(i_cells.shape[1]):
            z_arr =  opt_ik_numba_single(i_cells,j_genes,i,k,prob_matrix_a,prob_matrix_b,y,sigma_y,y_scale,z_arr,z_count)











