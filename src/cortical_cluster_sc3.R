library(dplyr)

fname = "~/scrpd/data/raw/GSE71585_RefSeq_counts.csv"

cortical_df = read.csv(fname)
rownames(cortical_df) =cortical_df$gene
cortical_df$gene <- NULL
cortical_df_t = as.data.frame(t(cortical_df))

library(SingleCellExperiment)
library(scran)
sce <- SingleCellExperiment(list(counts=as.matrix(cortical_df)))
clusters <- quickCluster(sce)
sce <- computeSumFactors(sce, cluster=clusters)
sce <- normalize(sce)
dim(logcounts(sce))
library(DrImpute)
cortical_exdata <- preprocessSC(logcounts(sce),min.expressed.cell=50, min.expressed.gene = 100)
library(umap)
cortical_umap = umap(t(cortical_exdata))
cortical_clust50 = kmeans(cortical_umap$layout,50)
cortical_clust100 = kmeans(cortical_umap$layout,100)
cortical_clust25 = kmeans(cortical_umap$layout,25)
cortical_clust10 = kmeans(cortical_umap$layout,10)
cortical_clustall = as.matrix(rbind(cortical_clust25$cluster,cortical_clust50$cluster,cortical_clust100$cluster,
                                    cortical_clust10$cluster))

cortical_imp <- DrImpute(cortical_exdata, cls=cortical_clustall)
rownames(cortical_imp)[2000]
colnames(cortical_imp)<-colnames(cortical_df)

library(diptest)
dip_pval <-function(vals){dip.test(vals)$p.value}
dip_vals= apply(cortical_imp,1,FUN=dip_pval)
hist(dip_vals)
cortical_imp_df = as.data.frame(cortical_imp[dip_vals<0.05/nrow(cortical_imp),])
write.csv(cortical_imp_df,"~/scrpd/data/processed/cortical_drimpute.csv")

