import numpy as np
import sys,math
import multiprocessing
from numba import float64,float32,jit,boolean,vectorize,int64
from numba.types import UniTuple, Tuple,void

from bimodal_mixture_model import make_bimodal_params_numba,value_to_bimodal_prob_numba
from cluster_init import assign_cells_init,spectral_cluster_cosine, ward_cluster_cosine, spectral_cluster,boolean_spectral_cluster, nmf_init, kmeans_cluster
from likelihood import full_log_likelihood_single,greedy_optimize_i_numba_single,greedy_optimize_j_numba_single,sgn
import random
import pickle
from sklearn import  model_selection

@jit(Tuple((float64[:],float64[:,:]))(float64[:,:],float64[:,:]),nopython=True)
def get_memo_params(i_cells:np.ndarray,j_genes:np.ndarray,):
    z_count = np.dot(i_cells,j_genes.T)
    z = sgn(z_count)
    z_arr = np.sum(z, 0)
    return z_arr,z_count

def estimate_epi_data(i_cells:np.ndarray,j_genes:np.ndarray):
    z = np.dot(i_cells, j_genes.T).astype(bool).astype(float)
    z_total = np.sum(z , 0) * (1.0 / (i_cells.shape[0] ))
    return z_total

def e_step(exp_data:np.ndarray, epi_data:np.ndarray):
    bimod_params = make_bimodal_params_numba(exp_data,epi_data)
    log_prob_a, log_prob_b = value_to_bimodal_prob_numba(exp_data, bimod_params)
    return log_prob_a,log_prob_b,bimod_params

def m_step(sample_clustering:np.ndarray,var_clustering:np.ndarray,
           log_prob_a: np.ndarray, log_prob_b:np.ndarray, epi_data:np.ndarray, sigma_y:float,
           y_scale:float):
    iteration = 0
    old_likehood = -1*sys.float_info.max
    likelihood = old_likehood/2.0
    while not(math.isclose(old_likehood,likelihood)) :
        print(iteration)
        old_likehood = likelihood
        iteration+=1
        z_arr, z_count,  = get_memo_params(sample_clustering, var_clustering)
        print("optmizing j")
        greedy_optimize_j_numba_single(sample_clustering, var_clustering,  log_prob_a, log_prob_b, epi_data,
                          sigma_y, y_scale,z_arr, z_count)
        print("optimizing i")
        greedy_optimize_i_numba_single(sample_clustering, var_clustering, log_prob_a, log_prob_b, epi_data,
                                sigma_y, y_scale, z_arr, z_count)
        likelihood = full_log_likelihood_single(sample_clustering,var_clustering,log_prob_a,log_prob_b,epi_data,sigma_y,y_scale)
        print(likelihood)
    return sample_clustering,var_clustering, likelihood


def bicluster_single(exp_data:np.ndarray, epi_data:np.ndarray, num_clusters:int,
                  sigma_y:float,y_scale:float):
    '''

    :param exp_data: N x G matrix of Single Cell Expression
    :param epi_data:  G length vector of gene-specific epigenomic measurements
    :param num_clusters: Number of clusters, K
    :param sigma_y: Noise of epigenomic data
    :param y_scale: weighting of epigenomic data in likelihood
    :return: Tuple(I,J,eps,epi_est): I is the  N x K Cell assignment matrix, J is the G:K Gene Assignment Matrix,
        eps is the bicluster population difference vector, epi_est is the estimate of the epigenomic vector
    '''
    #Estep using epi_data
    print("initial e-step")
    log_prob_a, log_prob_b, bimod_params = e_step(exp_data,epi_data)

    #print(np.sum(np.isnan(log_prob_a)))

    #Clustering initilizations
    print("initial clustering")
    gene_assignments = spectral_cluster_cosine(exp_data.T,num_clusters)
    #gene_assignments = kmeans_cluster(exp_data.T,num_clusters)
    cell_assignments = assign_cells_init(gene_assignments,log_prob_a,log_prob_b)
    likelihood= 0.0
    cell_assignments,gene_assignments,likelihood = m_step(cell_assignments,gene_assignments,log_prob_a,log_prob_b,epi_data,sigma_y,y_scale)
    cell_assignments_next = np.zeros(cell_assignments.shape)
    gene_assignments_next = np.zeros(gene_assignments.shape)
    iteration = 0
    while not (np.allclose(cell_assignments, cell_assignments_next) and np.allclose(gene_assignments, gene_assignments_next) or iteration>=20):
        if iteration>0:
            cell_assignments = np.copy(cell_assignments_next)
            gene_assignments = np.copy(gene_assignments_next)
        iteration+=1
        print(" e-step")
        epi_estimate = estimate_epi_data(cell_assignments, gene_assignments)
        log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_estimate)
        print("m-step")

        cell_assignments_next, gene_assignments_next,likelihood = m_step(cell_assignments, gene_assignments, log_prob_a,
                                                         log_prob_b, epi_data, sigma_y, y_scale)
    return cell_assignments,gene_assignments,epi_estimate, likelihood


def bicluster_preprocessed(exp_data:np.ndarray, epi_data:np.ndarray, num_clusters:int,
                  sigma_y:float,y_scale:float,log_prob_a, log_prob_b, bimod_params):
    '''

    :param exp_data: N x G matrix of Single Cell Expression
    :param epi_data:  G length vector of gene-specific epigenomic measurements
    :param num_clusters: Number of clusters, K
    :param sigma_y: Noise of epigenomic data
    :param y_scale: weighting of epigenomic data in likelihood
    :return: Tuple(I,J,eps,epi_est): I is the  N x K Cell assignment matrix, J is the G:K Gene Assignment Matrix,
        eps is the bicluster population difference vector, epi_est is the estimate of the epigenomic vector
    '''
    #Estep using epi_data
    #print("initial e-step")
    #log_prob_a, log_prob_b, bimod_params = e_step(exp_data,epi_data)

    #print(np.sum(np.isnan(log_prob_a)))

    #Clustering initilizations
    print("initial clustering")
    epi_estimate = np.copy(epi_data)

    gene_assignments = spectral_cluster_cosine(exp_data.T,num_clusters)
    #gene_assignments = nmf_init(exp_data.T,num_clusters)
    #gene_assignments = kmeans_cluster(exp_data.T,num_clusters)
    cell_assignments = assign_cells_init(gene_assignments,log_prob_a,log_prob_b)
    likelihood= 0.0
    cell_assignments,gene_assignments,likelihood = m_step(cell_assignments,gene_assignments,log_prob_a,log_prob_b,epi_data,sigma_y,y_scale)
    cell_assignments_next = np.zeros(cell_assignments.shape)
    gene_assignments_next = np.zeros(gene_assignments.shape)
    iteration = 0
    while not (np.allclose(cell_assignments, cell_assignments_next) and np.allclose(gene_assignments, gene_assignments_next) or iteration>=20):
        if iteration>0:
            cell_assignments = cell_assignments_next
            gene_assignments = gene_assignments_next
        iteration+=1
        print(" e-step")
        epi_estimate = estimate_epi_data(cell_assignments, gene_assignments)
        log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_estimate)
        print("m-step")

        cell_assignments_next, gene_assignments_next,likelihood = m_step(cell_assignments, gene_assignments, log_prob_a,
                                                         log_prob_b, epi_data, sigma_y, y_scale)
    return cell_assignments,gene_assignments,epi_estimate, likelihood



def bicluster_parallel(exp_data:np.ndarray, epi_data:np.ndarray, num_clusters:int,
                  sigma_y:float,y_scale:float,log_prob_a, log_prob_b, return_dict):
    '''

    :param exp_data: N x G matrix of Single Cell Expression
    :param epi_data:  G length vector of gene-specific epigenomic measurements
    :param num_clusters: Number of clusters, K
    :param sigma_y: Noise of epigenomic data
    :param y_scale: weighting of epigenomic data in likelihood
    :return: Tuple(I,J,eps,epi_est): I is the  N x K Cell assignment matrix, J is the G:K Gene Assignment Matrix,
        eps is the bicluster population difference vector, epi_est is the estimate of the epigenomic vector
    '''
    #Estep using epi_data
    #print("initial e-step")
    #log_prob_a, log_prob_b, bimod_params = e_step(exp_data,epi_data)

    #print(np.sum(np.isnan(log_prob_a)))

    #Clustering initilizations
    print("initial clustering")
    epi_estimate = np.copy(epi_data)

    gene_assignments = spectral_cluster_cosine(exp_data.T,num_clusters)
    #gene_assignments = nmf_init(exp_data.T,num_clusters)
    #gene_assignments = kmeans_cluster(exp_data.T,num_clusters)
    cell_assignments = assign_cells_init(gene_assignments,log_prob_a,log_prob_b)
    likelihood= 0.0
    cell_assignments,gene_assignments,likelihood = m_step(cell_assignments,gene_assignments,log_prob_a,log_prob_b,epi_data,sigma_y,y_scale)
    cell_assignments_next = np.zeros(cell_assignments.shape)
    gene_assignments_next = np.zeros(gene_assignments.shape)
    iteration = 0
    while not (np.allclose(cell_assignments, cell_assignments_next) and np.allclose(gene_assignments, gene_assignments_next) or iteration>=20):
        if iteration>0:
            cell_assignments = cell_assignments_next
            gene_assignments = gene_assignments_next
        iteration+=1
        print(" e-step")
        epi_estimate = estimate_epi_data(cell_assignments, gene_assignments)
        log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_estimate)
        print("m-step")

        cell_assignments_next, gene_assignments_next,likelihood = m_step(cell_assignments, gene_assignments, log_prob_a,
                                                         log_prob_b, epi_data, sigma_y, y_scale)
    return_dict[num_clusters] = (cell_assignments,gene_assignments,epi_estimate, likelihood)




def BIC(likelihood: float,num_cells: int,num_genes: int, num_datasets: int,num_clusters:int):
    return np.log(num_cells*num_genes)*(num_clusters*(num_cells+num_genes+num_datasets)+2*num_genes) - 2*likelihood

def AIC(likelihood: float,num_cells: int,num_genes: int, num_datasets: int,num_clusters:int):
    return 2*(num_clusters*(num_cells+num_genes+num_datasets)+2*num_genes) - 2*likelihood

def multi_k_em_parallel(exp_data:np.ndarray, epi_data:np.ndarray, cluster_range:range,
                  sigma_y:float,y_scale:float,ic:str):
    cluster_range = list(cluster_range)
    best_cell_assignments, best_gene_assignments, best_bicluster_assignments, best_bic, best_k = None,None,None,None, None
    # Estep using epi_data
    print("initial e-step")

    log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_data)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    for cluster_num in cluster_range:
        p = multiprocessing.Process(target=bicluster_parallel, args=(exp_data,epi_data,cluster_num,sigma_y,y_scale,log_prob_a,log_prob_b, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    ic_list =[]
    for cluster_num in cluster_range:
        if ic=="BIC":
            ic_list.append( BIC(return_dict[cluster_num][3],exp_data.shape[0],exp_data.shape[1],epi_data.size,cluster_num))
        elif ic=="AIC":
            ic_list.append( AIC(return_dict[cluster_num][3],exp_data.shape[0],exp_data.shape[1],epi_data.size,cluster_num))

        else:
            raise RuntimeError('Specified Information criterion does not exists')
    best_index = np.argmax(np.asarray(ic_list))
    best_cell_assignments, best_gene_assignments, epi_estimate, likelihood = return_dict[cluster_range[best_index]]


    return best_cell_assignments, best_gene_assignments, ic_list[best_index], cluster_range[best_index]



def multi_k_em(exp_data:np.ndarray, epi_data:np.ndarray, cluster_range:range,
                  sigma_y:float,y_scale:float,ic:str):
    best_cell_assignments, best_gene_assignments, best_bicluster_assignments, best_bic, best_k = None,None,None,None, None
    # Estep using epi_data
    print("initial e-step")

    log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_data)
    for cluster_num in cluster_range:
        cell_assignments, gene_assignments, epi_estimate,likelihood = bicluster_preprocessed(exp_data,epi_data,cluster_num,sigma_y,y_scale,log_prob_a,log_prob_b,bimod_params)
        if ic=="BIC":
            bic = BIC(likelihood,exp_data.shape[0],exp_data.shape[1],epi_data.size,cluster_num)
        elif ic=="AIC":
            bic = AIC(likelihood, exp_data.shape[0], exp_data.shape[1], epi_data.size, cluster_num)
        else:
            raise RuntimeError('Specified Information criterion does not exists')

        if best_bic is None or bic<best_bic:
            best_cell_assignments, best_gene_assignments,  best_bic, best_k =\
                cell_assignments,gene_assignments,bic,cluster_num
    return best_cell_assignments, best_gene_assignments, best_bic, best_k






