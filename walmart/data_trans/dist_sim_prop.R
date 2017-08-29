library(readr)
library(data.table)
library(proxy)
library(qlcMatrix)
library(cccd)
library(igraph)
library(bit64)
setwd("/media/SSHD1/kaggle/walmart")

# Load data
ts1Trans <- fread("/media/branden/SSHD1/kaggle/walmart/data_trans/ts1Trans3_abs.csv", header = TRUE)
# Department distance/similarity
ts1_dept <- as.matrix(ts1Trans[, 47:115, with=FALSE], nrow=nrow(ts1Trans))
ts1_dept_Matrix <- Matrix(ts1_dept)
ts1_cosSparse <- as.matrix(cosSparse(ts1_dept_Matrix))
ts1_dist <- pr_simil2dist(ts1_cosSparse)
ts1_dist_nng <- nng(dx=ts1_dist, k=4)

V(ts1_dist_nng)$name <- rownames(ts1_cosSparse)
E(ts1_dist_nng)$weight <- apply(get.edges(ts1_dist_nng,1:ecount(ts1_dist_nng)),1,function(x)ts1_cosSparse[x[1],x[2]])

ts1_dist_adj <- as_adjacency_matrix(ts1_dist_nng, attr="weight")
ts1_dist_adj_mat <- as.matrix(ts1_dist_adj)
dept_diag <- diag(x=1, nrow=nrow(ts1_dist_adj_mat))
ts1_dist_adj_mat <- ts1_dist_adj_mat + dept_diag

ts1_dist_adj_mat <- ts1_dist_adj_mat %*% diag(1/rowSums(ts1_dist_adj_mat))

ts1_dept_simil <- as.data.frame(ts1_dept %*% ts1_dist_adj_mat)
colnames(ts1_dept_simil) <- colnames(ts1_dept)

write_csv(ts1_dept_simil, "./data_trans/ts1_dept_simil_prop.csv")


# Fineline distance/similarity
ts1_fine <- as.matrix(ts1Trans[, 116:5469, with=FALSE], nrow=nrow(ts1Trans))
ts1_fine_Matrix <- Matrix(ts1_fine)
ts1_cosSparse <- as.matrix(cosSparse(ts1_fine_Matrix))
ts1_dist <- pr_simil2dist(ts1_cosSparse)
ts1_dist_nng <- nng(dx=ts1_dist, k=4)

V(ts1_dist_nng)$name <- rownames(ts1_cosSparse)
E(ts1_dist_nng)$weight <- apply(get.edges(ts1_dist_nng,1:ecount(ts1_dist_nng)),1,function(x)ts1_cosSparse[x[1],x[2]])

ts1_dist_adj <- as_adjacency_matrix(ts1_dist_nng, attr="weight")
ts1_dist_adj_mat <- as.matrix(ts1_dist_adj)
fine_diag <- diag(x=1, nrow=nrow(ts1_dist_adj_mat))
ts1_dist_adj_mat <- ts1_dist_adj_mat + fine_diag

ts1_dist_adj_mat <- ts1_dist_adj_mat %*% diag(1/rowSums(ts1_dist_adj_mat))

ts1_fine_simil <- as.data.frame(ts1_fine %*% ts1_dist_adj_mat)
colnames(ts1_fine_simil) <- colnames(ts1_fine)

write_csv(ts1_fine_simil, "./data_trans/ts1_fine_simil_prop.csv")

