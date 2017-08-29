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
t1 <- fread("./train.csv")
s1 <- fread("./test.csv")
# dest <- fread("./destinations.csv")

t1$id <- -1
s1 <- s1[,c("is_booking"):=1][,c("cnt", "hotel_cluster"):=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)


# Clusters are based in part on popularity
# May want to remove cases where there is only 1 combination. Else, may want to figure out how to prevent some possible overfitting
leak <- dcast.data.table(ts1[filter==0][1:31400000], user_location_country+user_location_region+user_location_city+hotel_market+orig_destination_distance ~ hotel_cluster, fun.aggregate=length, value.var="hotel_cluster")
leak <- merge(ts1[,c("id2","hotel_cluster","user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"),with=F], leak, by=c("user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"), all.x=TRUE, sort=FALSE)
clusterDum <- dummy(leak[1:31400000,hotel_cluster])
# clusterDum <- clusterDum[,2:101]
for (col in 1:100) {
  set(leak, i=1:31400000, j=col+7L, value=leak[[col+7L]][1:31400000]-clusterDum[,col+0L])
}
rm(clusterDum, t1, s1, ts1); gc()
rm(t1, s1, ts1)
leak2 <- data.table(leak[,id2], prop.table(as.matrix(leak[,8:ncol(leak), with=FALSE]),margin=1))
for (col in 2:ncol(leak2)) {
  set(leak2, j=col+0L, value=as.integer(leak2[[col+0L]]*10000000))
}
for (j in 2:ncol(leak2)){
  set(leak2, which(is.na(leak2[[j]])),j,-1)}
colnames(leak2) <- c("id2",paste("hotel_cluster_leak",colnames(leak2)[2:ncol(leak2)], sep="_"))

fwrite(leak2, "./data_trans/data_trans_leak_props.csv")
