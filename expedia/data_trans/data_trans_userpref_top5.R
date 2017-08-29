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
userpref <- dcast.data.table(ts1[filter==0], user_id + srch_destination_id + hotel_country + hotel_market~ hotel_cluster, fun.aggregate=function(x) length(x)*.15 + sum(x)*.85, value.var="is_booking")
userpref <- merge(ts1[,c("id2","filter","hotel_cluster","user_id","srch_destination_id","hotel_country","hotel_market"),with=F], userpref, by=c("user_id","srch_destination_id","hotel_country","hotel_market"), all.x=TRUE, sort=FALSE)
clusterDum <- dummy(userpref[filter %in% c(0,1),hotel_cluster])
# clusterDum <- clusterDum[,2:101]
for (col in 1:100) {
  set(userpref, i=1:nrow(clusterDum), j=col+7L, value=pmax(0,userpref[[col+7L]][1:nrow(clusterDum)]-clusterDum[,col+0L]))
}
rm(clusterDum, t1, s1, ts1); gc()
# rm(t1, s1, ts1); gc()
# userpref <- data.table(userpref[,id2], prop.table(as.matrix(userpref[,9:ncol(userpref), with=FALSE]),margin=1))
# for (col in 2:ncol(userpref)) {
#   set(userpref, j=col+0L, value=as.integer(userpref[[col+0L]]*10000000))
# }
# for (j in 2:ncol(userpref)){
#   set(userpref, which(is.na(userpref[[j]])),j,-1)}
# 
# # leak <- leak[,c(6,9:ncol(leak)),with=FALSE]
# colnames(leak) <- c("id2",paste("hotel_cluster_leak",colnames(leak)[2:ncol(leak)], sep="_"))



userprefTopK <- as.data.table(t(apply(userpref[,8:ncol(userpref),with=F], 1, function(x) names(sort(x[x>0], decreasing=T)[1:5]))))
colnames(userprefTopK)[1:ncol(userprefTopK)] <- paste0("userprefTop_",1:(ncol(userprefTopK)))
for (col in 1:ncol(userprefTopK)){
  set(userprefTopK, j=col, value = as.numeric(str_extract(userprefTopK[[col]], "[0-9]{1,2}$")))
}
for (col in 1:ncol(userprefTopK)){
  set(userprefTopK, i=which(is.na(userprefTopK[[col]])), j=col, value = -1)
}
userprefTopK <- cbind(id2=userpref$id2, userprefTopK)
setkey(userprefTopK, id2)


fwrite(userprefTopK, "./data_trans/data_trans_userpref_top5.csv")
