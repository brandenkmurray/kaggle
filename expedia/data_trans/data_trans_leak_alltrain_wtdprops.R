library(data.table)
library(doParallel)
library(igraph)
library(gtools)
library(dummies)
setwd("/media/branden/SSHD1/kaggle/expedia")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
# source("./data_trans/utils.R")

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

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
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),pred0=Mode(t1$hotel_cluster), dummy="A", filter=c(rep(0,31400000), rep(1,nrow(t1)-31400000), rep(2, nrow(s1))), ts1)



# Clusters are based in part on popularity
# May want to remove cases where there is only 1 combination. Else, may want to figure out how to prevent some possible overfitting
leak <- dcast.data.table(ts1[filter %in% c(0,1)], user_location_country+user_location_region+user_location_city+hotel_market+orig_destination_distance ~ hotel_cluster, fun.aggregate=function(x) length(x)*.15 + sum(x)*.85, value.var="is_booking")
leak <- merge(ts1[,c("id2","filter","hotel_cluster","user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"),with=F], leak, by=c("user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"), all.x=TRUE, sort=FALSE)
clusterDum <- dummy(leak[filter %in% c(0,1),hotel_cluster])
# clusterDum <- clusterDum[,2:101]
for (col in 1:100) {
  set(leak, i=1:nrow(clusterDum), j=col+8L, value=pmax(0,leak[[col+8L]][1:nrow(clusterDum)]-clusterDum[,col+0L]))
}
rm(clusterDum, t1, s1, ts1); gc()
# rm(t1, s1, ts1); gc()
leak <- data.table(leak[,id2], prop.table(as.matrix(leak[,9:ncol(leak), with=FALSE]),margin=1))
for (col in 2:ncol(leak)) {
  set(leak, j=col+0L, value=as.integer(leak[[col+0L]]*10000000))
}
for (j in 2:ncol(leak)){
  set(leak, which(is.na(leak[[j]])),j,-1)}

# leak <- leak[,c(6,9:ncol(leak)),with=FALSE]
colnames(leak) <- c("id2",paste("hotel_cluster_leak",colnames(leak)[2:ncol(leak)], sep="_"))
fwrite(leak, "./data_trans/data_trans_leak_alltrain_wtdprops.csv")
