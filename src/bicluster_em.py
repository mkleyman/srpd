import numpy as np
from typing import Tuple,List

from bimodal_mixture_model import make_bimodal_params_numba,value_to_bimodal_prob_numba
from cluster_init import assign_cells_init,spectral_cluster_cosine, ward_cluster_cosine, spectral_cluster,boolean_spectral_cluster
from likelihood import greedy_optimize_i,greedy_optimize_j, eps_likelihood, full_log_likelihood, y_log_likelihood, greedy_optimize_i_numba, greedy_optimize_j_numba
import random
import pickle
from sklearn import  model_selection


def get_memo_params(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray):
    z_count = np.dot(i_cells,j_genes.T)
    z = z_count.astype(bool).astype(float)
    i_eps = np.zeros(i_cells.shape)
    for k in range(i_cells.shape[1]):
        i_eps[:, k] = eps[k] * i_cells[:, k]
    z_eps = np.dot(i_eps, j_genes.T)
    z_arr = np.sum(z_eps+z, 0)
    denom = np.sum(i_eps)+i_cells.shape[0]
    return z_arr,z_count,denom

def estimate_epi_data(i_cells:np.ndarray,j_genes:np.ndarray,eps:np.ndarray):
    z = np.dot(i_cells, j_genes.T).astype(bool).astype(float)
    i_eps = np.zeros(i_cells.shape)
    for k in range(i_cells.shape[1]):
        i_eps[:, k] = eps[k] * i_cells[:, k]
    z_eps = np.dot(i_eps, j_genes.T)
    z_total = np.sum(z + z_eps, 0) * (1 / (i_cells.shape[0] + np.sum(i_eps)))
    return z_total

def e_step(exp_data:np.ndarray, epi_data:np.ndarray)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    bimod_params = make_bimodal_params_numba(exp_data,epi_data)
    log_prob_a, log_prob_b = value_to_bimodal_prob_numba(exp_data, bimod_params)
    return log_prob_a,log_prob_b,bimod_params

def m_step(sample_clustering:np.ndarray,var_clustering:np.ndarray,eps:np.ndarray,
           log_prob_a: np.ndarray, log_prob_b:np.ndarray, epi_data:np.ndarray, sigma_y:float,
           y_scale:float):
    var_clustering_next = np.zeros(var_clustering.shape)
    sample_clustering_next = np.zeros(sample_clustering.shape)

    iteration = 0
    while not(np.allclose(sample_clustering,sample_clustering_next) and np.allclose(var_clustering,var_clustering_next)):
        print(iteration)

        if iteration>0:
            sample_clustering = np.copy(sample_clustering_next)
            var_clustering = np.copy(var_clustering_next)
        iteration+=1
        var_clustering_next = np.copy(var_clustering)
        sample_clustering_next = np.copy(sample_clustering)
        #z_arr, z_count, denom = get_memo_params(sample_clustering_next,var_clustering_next,eps)

        z_arr, z_count, denom = get_memo_params(sample_clustering_next, var_clustering_next, eps)

        print("optmizing j")
        #greedy_optimize_j(sample_clustering_next,var_clustering_next,eps_next,log_prob_a,log_prob_b,epi_data,sigma_y,sigma_eps,y_scale,
                          #z_arr, z_count, denom)
        greedy_optimize_j_numba(sample_clustering_next, var_clustering_next, eps, log_prob_a, log_prob_b, epi_data,
                          sigma_y, y_scale,z_arr, z_count, denom)

        '''


        z_arr, z_count, denom = get_memo_params(sample_clustering_next, var_clustering_next, eps_min)'''

        print("optimizing i")
        #greedy_optimize_i(sample_clustering_next, var_clustering_next, eps_next, log_prob_a, log_prob_b, epi_data,
                          #sigma_y, sigma_eps, y_scale,z_arr, z_count, denom)
        greedy_optimize_i_numba(sample_clustering_next, var_clustering_next, eps, log_prob_a, log_prob_b, epi_data,
                                sigma_y, y_scale, z_arr, z_count, denom)




        print(full_log_likelihood(sample_clustering_next,var_clustering_next,log_prob_a,log_prob_b,epi_data,sigma_y,sigma_eps,y_scale))
    return sample_clustering,var_clustering,eps








def bicluster(exp_data:np.ndarray, epi_data:np.ndarray, num_clusters:int,
                  sigma_y:float,sigma_eps:float,y_scale:float, opt:str, eps_opt:bool)->Tuple[np.ndarray,np.ndarray]:
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

    #eps = np.ones(num_clusters)*.1





    cell_assignments,gene_assignments, eps = m_step(cell_assignments,gene_assignments,eps,log_prob_a,log_prob_b,epi_data,sigma_y,sigma_eps,y_scale,eps_opt)
    print(eps)


    cell_assignments_next = np.zeros(cell_assignments.shape)
    gene_assignments_next = np.zeros(gene_assignments.shape)
    iteration = 0
    while not (np.allclose(cell_assignments, cell_assignments_next) and np.allclose(gene_assignments, gene_assignments_next) or iteration>=20):
        if iteration>0:
            cell_assignments = cell_assignments_next
            gene_assignments = gene_assignments_next
        iteration+=1
        print(" e-step")
        epi_estimate = estimate_epi_data(cell_assignments, gene_assignments, eps)
        log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_estimate)
        print("m-step")
        if opt == "grasp":
            cell_assignments_next, gene_assignments_next, eps = m_step_grasp(cell_assignments, gene_assignments, eps,
                                                                       log_prob_a,
                                                                       log_prob_b, epi_data, sigma_y, sigma_eps,
                                                                       y_scale)
        else:
            cell_assignments_next, gene_assignments_next,eps = m_step(cell_assignments, gene_assignments, eps, log_prob_a,
                                                         log_prob_b, epi_data, sigma_y, sigma_eps, y_scale,eps_opt)
    return cell_assignments,gene_assignments,eps,epi_estimate


def find_best_k_quick(exp_data:np.ndarray, epi_data:np.ndarray, range_clusters, sigma_y, sigma_eps, pickle_file:str):


    held_out_indices = random.sample(range(exp_data.shape[0]), int(exp_data.shape[0]/5.0))
    kept_in_indices = list(set(range(exp_data.shape[0])) - set(held_out_indices))
    held_out_data = exp_data[held_out_indices]
    kept_in_data = exp_data[kept_in_indices]

    best_likelihood = None
    best_k = None

    optimal_params = []

    for k in range_clusters:
        print("running for cluster num:")
        print(k)
        print("optimizing training ")
        sample_clustering, var_clustering, eps, epi_estimate = bicluster(kept_in_data,epi_data,k,sigma_y,sigma_eps,exp_data.shape[0],"greedy")
        params = (k,sample_clustering, var_clustering, eps, epi_estimate)
        optimal_params.append((params))
        pickle.dump(optimal_params,open( pickle_file, "wb" ) )
        print("performing likelihood evaluation on left out ")

        prob_matrix_a, prob_matrix_b, bimod_params = e_step(held_out_data, epi_estimate)
        i_init = assign_cells_init(var_clustering, prob_matrix_a, prob_matrix_b)
        z_arr, z_count, denom = get_memo_params(i_init, var_clustering, eps)

        greedy_optimize_i_numba(i_init, var_clustering, eps, prob_matrix_a, prob_matrix_b, epi_data,
                                sigma_y, exp_data.shape[0], z_arr, z_count, denom)
        held_out_likelihood = full_log_likelihood(i_init,var_clustering,eps,prob_matrix_a,prob_matrix_b,epi_data,sigma_y,sigma_eps,exp_data.shape[0])

        print("Likelihood:")
        print(held_out_likelihood)

        if best_likelihood is None or held_out_likelihood < best_likelihood:
            best_k = k
            best_likelihood = held_out_likelihood
    print("Best k:")
    print(best_k)

    sample_clustering, var_clustering, eps, epi_estimate = bicluster(exp_data,epi_data,best_k,sigma_y,sigma_eps,exp_data.shape[0],"greedy")
    params = (sample_clustering, var_clustering, eps, epi_estimate)
    optimal_params.append((params))
    pickle.dump(optimal_params, open(pickle_file, "wb"))
    return params


def find_best_k_full(exp_data:np.ndarray, epi_data:np.ndarray, range_clusters, sigma_y, sigma_eps, pickle_file:str,eps_opt:bool):




    best_likelihood = None
    best_k = None

    kf = model_selection.KFold(n_splits=4)

    optimal_params = []

    for k in range_clusters:
        k_likelihood = 0
        for train_index, test_index in kf.split(exp_data):
            print("running for cluster num:")
            print(k)
            print("optimizing training ")
            sample_clustering, var_clustering, eps, epi_estimate= bicluster(exp_data[train_index],epi_data,k,sigma_y,sigma_eps,exp_data.shape[0],"greedy",eps_opt)
            params = (k,sample_clustering, var_clustering, eps, epi_estimate)
            optimal_params.append((params))
            pickle.dump(optimal_params,open( pickle_file, "wb" ) )
            print("performing likelihood evaluation on left out ")
            adjustment = min([0.0, np.amin( epi_estimate)])
            epi_estimate +=   (-1 * adjustment)

            prob_matrix_a, prob_matrix_b, bimod_params = e_step(exp_data[test_index], epi_estimate)
            i_init = assign_cells_init(var_clustering, prob_matrix_a, prob_matrix_b)
            z_arr, z_count, denom = get_memo_params(i_init, var_clustering, eps)

            greedy_optimize_i_numba(i_init, var_clustering, eps, prob_matrix_a, prob_matrix_b, epi_data,
                                    sigma_y, exp_data.shape[0], z_arr, z_count, denom)
            held_out_likelihood = full_log_likelihood(i_init,var_clustering,eps,prob_matrix_a,prob_matrix_b,epi_data,sigma_y,sigma_eps,exp_data.shape[0])

            print("Likelihood:")
            print(held_out_likelihood)
            k_likelihood+=held_out_likelihood

        if best_likelihood is None or k_likelihood < best_likelihood:
            best_k = k
            best_likelihood = k_likelihood
    print("Best k:")
    print(best_k)

    sample_clustering, var_clustering, eps, epi_estimate = bicluster(exp_data,epi_data,best_k,sigma_y,sigma_eps,exp_data.shape[0],"greedy",eps_opt)
    params = (sample_clustering, var_clustering, eps, epi_estimate)
    optimal_params.append((params))
    pickle.dump(optimal_params, open(pickle_file, "wb"))
    return params

def BIC(likelihood: float,num_cells: int,num_genes: int, num_datasets: int,num_clusters:int):
    return np.log(num_cells*num_genes)*(num_clusters*(num_cells+num_genes+num_datasets)+2*num_genes) - 2*likelihood

def AIC(likelihood: float,num_cells: int,num_genes: int, num_datasets: int,num_clusters:int):
    return 2*(num_clusters*(num_cells+num_genes+num_datasets)+2*num_genes) - 2*likelihood

def multi_k_em(exp_data:np.ndarray, epi_data:np.ndarray, cluster_range:range,
                  sigma_y:float,y_scale:float,ic:str):
    best_cell_assignments, best_gene_assignments, best_bicluster_assignments, best_bic, best_k = None,None,None,None, None
    # Estep using epi_data
    print("initial e-step")

    log_prob_a, log_prob_b, bimod_params = e_step(exp_data, epi_data)
    for cluster_num in cluster_range:
        cell_assignments, gene_assignments, bicluster_assignments, likelihood =\
            bicluster(exp_data,epi_data,cluster_num,sigma_y,y_scale,log_prob_a,log_prob_b)
        if ic=="BIC":
            bic = BIC(likelihood,exp_data.shape[0],exp_data.shape[1],epi_data.shape[1],cluster_num)
        elif ic=="AIC":
            bic = AIC(likelihood, exp_data.shape[0], exp_data.shape[1], epi_data.shape[1], cluster_num)
        else:
            raise RuntimeError('Specified Information criterion does not exists')

        if best_bic is None or bic<best_bic:
            best_cell_assignments, best_gene_assignments, best_bicluster_assignments, best_bic, best_k = \
                cell_assignments,gene_assignments,bicluster_assignments,bic,cluster_num
    return best_cell_assignments, best_gene_assignments, best_bicluster_assignments, best_bic, best_k






