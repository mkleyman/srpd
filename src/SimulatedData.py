import random
from typing import List
import numpy as np
from scipy.special import expit



class SimBicluster:

    def __init__(self,cluster_id, num_vars: int, num_chosen: int):
        '''

        :param cluster_id: unique id for each cluster
        :param num_vars:  number of genes in dataset
        :param num_chosen: number of genes in bicluster
        '''
        population = list( range(num_vars))
        self.vars = set(random.sample(population,num_chosen))
        self.samples =  set()
        self.num_vars = num_vars
        self.cluster_id = cluster_id
        #cluster size noise


def create_sample_easy(sample_id, num_vars:int, bicluster_list:List[SimBicluster],epi_data: np.ndarray,
                       num_tfs:int, epi_error:float, error_type:str):
    '''

    :param sample_id: Sample unique id
    :param num_vars: number of genes
    :param bicluster_list: list of bicluster objects
    :param epi_data: epigenetic gene vector
    :param num_tfs: number of bicluters the sample belongs too
    :param epi_error: error in epigentic measurements
    :return: sample vector of gene measurements
    '''

    #randomly choose biclusters that the
    chosen_clusters = random.sample(bicluster_list,num_tfs)
    chosen_vars = set()
    #union set of genes that are activated in sample
    for cluster in chosen_clusters:
        chosen_vars.update(cluster.vars)
        cluster.samples.add(sample_id)

    #sample = [random.normalvariate(5,1) if i in chosen_vars else random.normalvariate(0,1)
              #for i in range(num_vars)]
    if error_type == "mode":
        state_row = [np.random.binomial(1, 1 - epi_error) if i in chosen_vars else np.random.binomial(1, epi_error)
                   for i in range(num_vars)]
        sample =[random.normalvariate(3, 1) if state_row[i]==1 else random.normalvariate(0, 1)
                 for i in range(num_vars)]
    else:
        sample = [random.normalvariate(3,epi_error) if i in chosen_vars else random.normalvariate(0,epi_error)
            for i in range(num_vars)]

    #update the epigenetic vector with noise
    epi_row = [1.0 if i in chosen_vars else 0.0 for i in range(num_vars)]
    #epi_row = [np.random.binomial(1, 1 - epi_error) if i in chosen_vars else np.random.binomial(1, epi_error)
               #for i in range(num_vars)]
    #epi_row = [1 if i in chosen_vars else 0 for i in range(num_vars)]
    epi_data.__iadd__(epi_row)
    return sample


def create_sample_dropout(sample_id, num_vars:int, bicluster_list:List[SimBicluster],epi_data: np.ndarray,
                       num_clusters:int, epi_error:float, prob_drop_out:float):

    chosen_clusters = random.sample(bicluster_list,num_clusters)
    chosen_vars = set()
    for cluster in chosen_clusters:
        chosen_vars.update(cluster.vars)
        cluster.samples.add(sample_id)
    sample_clean = [random.normalvariate(10,1) if i in chosen_vars else random.normalvariate(0,1)
              for i in range(num_vars)]
    sample = [sample_var if random.random()<prob_drop_out else 0 for sample_var in sample_clean]
    epi_row = [np.random.binomial(1,1-epi_error) if i in chosen_vars else np.random.binomial(1,epi_error)
               for i in range(num_vars)]
    epi_data.__iadd__(epi_row)
    return sample


def create_data(num_vars:int,num_samples:int, num_clusters:int, vars_per_cluster:int, epi_error:float,error_type:str):
    '''
    :param num_vars: Number of genes in the dataset
    :param num_samples: Number of samples in the data set
    :param num_clusters: Number of biclusters
    :param vars_per_cluster:  Number of genes per bicluster
    :param epi_error: Noise in epigenetic data
    :return: Tuple of size: List of biclusters, samples x genes expression matrix, epigenetic vectors
    '''
    cluster_list = []
    #initialize biclusters
    chosen_vars =  set()
    for cluster_id in range(num_clusters):
        bicluster = SimBicluster(cluster_id,num_vars,vars_per_cluster)
        cluster_list.append(bicluster)
        chosen_vars.update(bicluster.vars)

    #deal unchosen vars

    all_vars =  set(range(num_vars))
    unchosen_vars = all_vars - chosen_vars
    for var in unchosen_vars:
        rnd_cluster = cluster_list[random.randrange(num_clusters)]
        rnd_cluster.vars.add(var)
        rnd_cluster.num_vars = rnd_cluster.num_vars +1

    #initialize samples
    sample_list = []
    #epigenetic data initialized as vector of zeros
    epi_data = np.zeros(num_vars)
    for sample_id in range(num_samples):
        #Sample can come from 1 to 3 biclusters
        num_tfs = random.randrange(1,3)
        sample = create_sample_easy(sample_id,num_vars, cluster_list, epi_data,num_tfs,epi_error, error_type )
        sample_list.append(sample)
    sample_data = np.asarray(sample_list)
    epi_data = epi_data/num_samples
    #add epigenetic noise

    return(cluster_list,sample_data, epi_data)

def bicluster_tuple_maker(bicluster_list,num_samples,num_vars):
    '''

    :param bicluster_list: List of biclusters objects
    :param num_samples: Number of cells in the dataset
    :param num_vars: Number of genes in the datset
    :return: tuple of of: b x n matrix of biclusters by sample membership, b x g matrix of bicluster by gene membership
    '''
    rows_ = np.zeros((len(bicluster_list),num_samples))
    columns_ = np.zeros((len(bicluster_list),num_vars))
    for b_index in range(len(bicluster_list)):
        for sample in range(num_samples):
            if sample in bicluster_list[b_index].samples:
                rows_[b_index,sample]=1.0
        for var in range(num_vars):
            if var in bicluster_list[b_index].vars:
                columns_[b_index,var]=1.0
    return rows_.astype(bool),columns_.astype(bool)

def epi_matrix_maker(bicluster_rows:np.ndarray,bicluster_cols:np.ndarray):
    '''

    :param bicluster_rows:  n x b matrix of biclusters by sample membership
    :param bicluster_cols: g x b matrix of bicluster by gene membership
    :return:
    '''
    return np.dot(bicluster_rows.astype(int).T, bicluster_cols.astype(int)).astype(bool).astype(float)


if __name__ == "__main__":
    bicluster_list,sample_data,epi_data = create_data(1000,1000,6,500,.1)
    print(epi_data)
    print(bicluster_tuple_maker(bicluster_list,1000,1000))






