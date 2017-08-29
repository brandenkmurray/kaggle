library(data.table)
library(doParallel)
library(igraph)
library(gtools)
library(dummies)
library(caret)
library(bigpca)
library(bigalgebra)
# setwd("/media/branden/SSHD1/kaggle/expedia")
setwd("~/ebs")
threads <- detectCores() - 2

#######################
## Load data
#######################
t1 <- fread("./train.csv", select=c("srch_destination_id"))
s1 <- fread("./test.csv", select=c("srch_destination_id"))
# dest <- fread("./destinations.csv")

ts1 <- rbind(t1, s1)
# Combine into 1 data frame
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)
rm(t1,s1)

dest <- fread("./destinations.csv")
ts1 <- merge(ts1, dest, by="srch_destination_id", all.x=TRUE, sort=FALSE)
id2 <- ts1$id2
for (j in 4:ncol(ts1)){
  set(ts1, which(is.na(ts1[[j]])),j,median(ts1[[j]], na.rm=TRUE))}



# get.col <- function(i) x[,i] # Emulates reading a column
# 
# xt.x <- matrix(numeric(), n, n)
# x.means <- rep(numeric(), n)
# for (i in 1:n) {
#   i.col <- get.col(i)
#   x.means[i] <- mean(i.col)
#   xt.x[i,i] <- sum(i.col * i.col)
#   if (i < n) {
#     for (j in (i+1):n) {
#       j.col <- get.col(j)
#       xt.x[i,j] <- xt.x[j,i] <- sum(j.col * i.col)
#     }    
#   }
# }
# xt.x <- (xt.x - m * outer(x.means, x.means, `*`)) / (m-1)
# svd.0 <- svd(xt.x / m)




bm <- as.big.matrix(ts1[,4:ncol(ts1),with=FALSE])

rm(ts1);gc()

bm_pca <- big.PCA(bm, pcs.to.keep = 60)
gc()
m <- as.matrix(bm)
rm(bm);gc()
x <- matrix(0,nrow = nrow(m), ncol=ncol(bm_pca$PCs))
for (i in 1:ncol(bm_pca$PCs)){
  x[,i] <- m %*% bm_pca$PCs[,i]
  }



ts1_pca <- cbind(id2, x)
ts1_pca <- as.data.frame(ts1_pca)
colnames(ts1_pca) <- c("id2", paste0("PCA_",1:60))

for (col in 1:100) {
  set(dest, j=col+1L, value=as.integer(dest[[col+1L]]*1000000))
}

fwrite(ts1_pca, "./data_trans/data_trans_pca.csv")
