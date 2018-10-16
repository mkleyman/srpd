import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from scipy.spatial.distance import squareform,pdist
from scipy.stats import hypergeom
from sklearn.cluster.bicluster import   SpectralBiclustering, SpectralCoclustering
from sklearn.manifold import SpectralEmbedding
import nimfa
import itertools



def num_to_bin(num:int):
    return np.frombuffer(np.binary_repr(num).encode('ascii'), dtype=bool)


def init_clusts_chip(epi_arr, thresh):
    tf_combination_dict = {}
    used_gene_set = set()
    gene_thresh = thresh*epi_arr.shape[0]
    gene_ids = np.arange(epi_arr.shape[0])
    tf_ids = list(range(epi_arr.shape[1]))
    for num_tfs in range(epi_arr.shape[1],0,-1):
        for tf_combination in itertools.combinations(tf_ids,num_tfs):
            gene_selection = np.sum(epi_arr[:,tf_combination]>0.0,1) == num_tfs
            genes = gene_ids[gene_selection]
            remaining_genes = set(genes).difference(used_gene_set)
            if len(remaining_genes)>gene_thresh:
                used_gene_set = used_gene_set.union(remaining_genes)
                tf_combination_dict[tuple(tf_combination)] = np.asarray(list(remaining_genes))
    num_clusters = len(tf_combination_dict)
    tf_matrix = np.zeros(shape=(epi_arr.shape[1],num_clusters))
    gene_matrix = np.zeros(shape=(epi_arr.shape[0],num_clusters))
    cluster = 0
    for tf,genes in tf_combination_dict.items():
        tf_matrix[list(tf),cluster] = 1.0
        gene_matrix[genes,cluster] = 1.0
        cluster += 1

    return tf_matrix, gene_matrix


def nmf_init(mat:np.ndarray, num_clusters=int):
    nmf = nimfa.Snmf(mat, rank=num_clusters)
    nmf_fit = nmf()
    return nmf_fit.basis().astype(bool).astype(float)

def init_hybrid(exp_mat,epi_mat,thresh,nclust):
    num_components = epi_mat.shape[1]-1
    corr_dist =  1.0-squareform(pdist(exp_mat.transpose(),metric="cosine"))
    spec = SpectralEmbedding(n_components=num_components, affinity="precomputed")
    spec_coord = spec.fit_transform(corr_dist)
    epi_bin = (epi_mat>0.0).astype(float)
    full_coord = np.concatenate((spec_coord,epi_bin),1)
    kmeans = KMeans(n_clusters = nclust).fit(full_coord)
    tf_matrix = (kmeans.cluster_centers_[:,num_components:]>thresh).astype(float).transpose()
    gene_matrix = binarize_vector(kmeans.labels_,nclust)
    return tf_matrix,gene_matrix









def binarize_vector(arr:np.ndarray, num_clusters):
    bin_arr = np.zeros((arr.size,num_clusters))
    bin_arr[np.arange(arr.size),arr] = 1
    return bin_arr

def assign_cells_init(gene_bin_mat:np.ndarray, prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray):
    inv_bin = 1.0 - gene_bin_mat
    one_bin = np.ones(gene_bin_mat.shape)
    prob_a = np.dot(prob_matrix_a,gene_bin_mat)+np.dot(prob_matrix_b,inv_bin)
    prob_b = np.dot(prob_matrix_b,one_bin)
    return (prob_a >  prob_b).astype(float)

def assign_genes_init(cell_bin_mat:np.ndarray, prob_matrix_a: np.ndarray, prob_matrix_b: np.ndarray):
    inv_bin = 1.0 - cell_bin_mat
    prob_a = np.dot(prob_matrix_a,cell_bin_mat)+np.dot(prob_matrix_b,inv_bin)
    prob_b = np.dot(prob_matrix_b,cell_bin_mat)+np.dot(prob_matrix_a,inv_bin)
    return (prob_a >  prob_b).astype(float)

def kmeans_cluster(data:np.ndarray, num_clusters:int):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    return binarize_vector(kmeans.labels_,num_clusters)

def spec_bi_cluster(data:np.ndarray, num_clusters:int):
    kmeans = SpectralBiclustering(n_clusters=num_clusters)
    kmeans.fit(data)
    return binarize_vector(kmeans.row_labels_,num_clusters)

def spec_co_cluster(data:np.ndarray, num_clusters:int):
    kmeans = SpectralCoclustering(n_clusters=num_clusters)
    kmeans.fit(data)
    return binarize_vector(kmeans.row_labels_,num_clusters)

def spectral_cluster(data:np.ndarray, num_clusters:int):
    spec = SpectralClustering(n_clusters=num_clusters)
    spec.fit(data)
    return binarize_vector(spec.labels_,num_clusters)

def spectral_cluster_combined(data:np.ndarray,epi_data:np.ndarray, num_clusters:int):
    #dist = squareform(pdist(data,'correlation'))*squareform(pdist(epi_data,'canberra'))
    dist = squareform(pdist(epi_data,'canberra'))
    spec = SpectralClustering(n_clusters=num_clusters,affinity="precomputed")
    spec.fit(dist)
    return binarize_vector(spec.labels_,num_clusters)

def spectral_cluster_cosine(data:np.ndarray, num_clusters:int):
    spec = SpectralClustering(n_clusters=num_clusters,affinity="cosine")
    spec.fit(data)
    return binarize_vector(spec.labels_,num_clusters)

def ward_cluster_cosine(data:np.ndarray, num_clusters:int):
    spec = AgglomerativeClustering(n_clusters=num_clusters,affinity="cosine",linkage="average")
    spec.fit(data)
    return binarize_vector(spec.labels_,num_clusters)

def boolean_spectral_cluster(log_prob_matrix_a:np.ndarray,log_prob_matrix_b:np.ndarray, num_clusters:int, distance:str):
    log_diff_matrix = log_prob_matrix_a > log_prob_matrix_b
    dist_mat = squareform(pdist(log_diff_matrix.T, metric=distance))
    spec = SpectralClustering(n_clusters=num_clusters, affinity="precomputed")
    spec.fit(dist_mat)
    return binarize_vector(spec.labels_,num_clusters)


def tf_kmeans(epi_data_tf):
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(np.matrix(epi_data_tf).transpose())
    return kmeans.labels_

def tf_hyper_geom(selected_genes:np.ndarray,epi_data_clustered):
    M= selected_genes.size
    N = np.sum(epi_data_clustered)
    n = np.sum(selected_genes)
    k = np.sum(selected_genes*epi_data_clustered)-1
    pval = hypergeom.sf(k,M,n,N)
    return pval<0.1

def b_init(j_genes:np.ndarray,epi_data:np.ndarray):
    bicluster_matrix = np.zeros(shape=(epi_data.shape[1],j_genes.shape[1]))
    for d in range(epi_data.shape[1]):
        epi_data_clustered = tf_kmeans(epi_data[:,d])
        for k in range(j_genes.shape[1]):
            if tf_hyper_geom(j_genes[:,k],epi_data_clustered): bicluster_matrix[d,k] = 1.0
    return bicluster_matrix


