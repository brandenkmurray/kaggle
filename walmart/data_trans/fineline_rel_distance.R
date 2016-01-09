library(readr)
library(data.table)
library(proxy)
library(qlcMatrix)
library(cccd)
setwd("/home/branden/Documents/kaggle/walmart")


ts1Trans <- data.table(read_csv("./data_trans/ts1Trans3_prop.csv"))
ts1_fine <- as.matrix(ts1Trans[, 47:116, with=FALSE], nrow=nrow(ts1Trans))
ts1_fine_Matrix <- Matrix(ts1_fine)
ts1_cosSparse <- cosSparse(ts1_fine_Matrix)

ts1_simil <- simil(ts1_fine, method="cosine" ,by_rows=FALSE)
ts1_simil_mat <- as.matrix(ts1_simil, diag=1)

ts1_dist <- dist(ts1_fine, method="cosine", by_rows=FALSE)
ts1_dist_matrix <- as.matrix(ts1_dist, diag=0)

ts1_fine_simil_diffuse <- as.data.frame(ts1_fine %*% ts1_simil_mat)
ts1_fine_dist_diffuse <- as.data.frame(ts1_fine %*% ts1_dist_matrix)

write_csv(ts1_fine_simil_diffuse, "./data_trans/ts1_fine_simil_diffuse_rel.csv")
write_csv(ts1_fine_dist_diffuse, "./data_trans/ts1_fine_dist_diffuse_rel.csv")
