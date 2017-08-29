library(data.table)
library(doParallel)
library(igraph)
library(gtools)
library(dummies)
setwd("/media/branden/SSHD1/kaggle/expedia")
threads <- detectCores() - 2

#######################
## Load data
#######################
t1 <- fread("./train.csv", select=c("srch_destination_id","hotel_cluster"))
s1 <- fread("./test.csv", select=c("srch_destination_id"))
# dest <- fread("./destinations.csv")

s1$hotel_cluster <- -1
ts1 <- rbind(t1, s1)
# Combine into 1 data frame
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)
rm(t1,s1)


# Clusters are based in part on popularity
# May want to remove cases where there is only 1 combination. Else, may want to figure out how to prevent some possible overfitting
srch_destid_clust_cast <- dcast.data.table(ts1[ts1$filter==0,], srch_destination_id ~ hotel_cluster, fun.aggregate=length, value.var="hotel_cluster")
srch_destid_clust_cast <- merge(ts1[,c("id2","hotel_cluster","srch_destination_id"),with=F], srch_destid_clust_cast, by=c("srch_destination_id"), all.x=TRUE, sort=FALSE)
clusterDum <- dummy(srch_destid_clust_cast[,hotel_cluster])
clusterDum <- clusterDum[,-1]


for (col in 1:100) {
  set(srch_destid_clust_cast, j=col+3L, value=srch_destid_clust_cast[[col+3L]]-clusterDum[,col+0L])
}
rm(clusterDum, ts1); gc()
srch_destid_clust_cast2 <- data.table(srch_destid_clust_cast[,id2], prop.table(as.matrix(srch_destid_clust_cast[,4:ncol(srch_destid_clust_cast), with=FALSE]),margin=1))
for (col in 2:ncol(srch_destid_clust_cast2)) {
  set(srch_destid_clust_cast2, j=col+0L, value=as.integer(srch_destid_clust_cast2[[col+0L]]*100000000))
}
for (j in 2:ncol(srch_destid_clust_cast2)){
  set(srch_destid_clust_cast2, which(is.na(srch_destid_clust_cast2[[j]])),j,-1)}
colnames(srch_destid_clust_cast2) <- c("id2",paste("hotel_clust_destid",colnames(srch_destid_clust_cast2)[2:ncol(srch_destid_clust_cast2)], sep="_"))

fwrite(srch_destid_clust_cast2, "./data_trans/data_trans_destid_props.csv")
