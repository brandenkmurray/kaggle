library(data.table)
library(RankAggreg)
library(doParallel)
library(doRNG)
bestpub <- fread("./stack_models/L1/testPreds/best_public_submission_2016-05-30-15-44.csv", header=TRUE)
# xgb16 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v16.csv", header=TRUE)
# xgb17 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v17.csv", header=TRUE)
# xgb18 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v18.csv", header=TRUE)
# xgb19 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v19.csv", header=TRUE)
# xgb20 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v20.csv", header=TRUE)
# xgb21 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v21_probs.csv", header=TRUE)
xgb1_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_loop.csv", header=TRUE)
xgb2_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb2_loop.csv", header=TRUE)
xgb3_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb3_loop.csv", header=TRUE)
xgb5_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb5_loop.csv", header=TRUE)

# xgb21 <- as.data.frame(t(apply(xgb21[,2:101,with=F], 1, function(x) names(sort(x, decreasing=T)[1:5]))))
# xgb21 <- do.call("paste", c(xgb21, sep=" "))
# xgb21 <- data.table(id=seq(0,length(xgb21)-1,1), hotel_cluster=xgb21)

pub_split <- data.frame(do.call('rbind', strsplit(as.character(bestpub$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb16_split <- data.frame(do.call('rbind', strsplit(as.character(xgb16$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb17_split <- data.frame(do.call('rbind', strsplit(as.character(xgb17$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb18_split <- data.frame(do.call('rbind', strsplit(as.character(xgb18$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb19_split <- data.frame(do.call('rbind', strsplit(as.character(xgb19$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb20_split <- data.frame(do.call('rbind', strsplit(as.character(xgb20$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
# xgb21_split <- data.frame(do.call('rbind', strsplit(as.character(xgb21$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
xgb1_loop_split <- data.frame(do.call('rbind', strsplit(as.character(xgb1_loop$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
xgb2_loop_split <- data.frame(do.call('rbind', strsplit(as.character(xgb2_loop$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
xgb3_loop_split <- data.frame(do.call('rbind', strsplit(as.character(xgb3_loop$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
xgb5_loop_split <- data.frame(do.call('rbind', strsplit(as.character(xgb5_loop$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)

# locsplit <- data.frame(do.call('rbind', strsplit(as.character(bestloc$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)


# cl <- makeCluster(14)
# registerDoParallel(cl)
# raVec <- foreach(i=1:nrow(bestloc), .combine=c, .packages=c("data.table", "RankAggreg")) %dopar% {
#    paste(RankAggreg(matrix(c(as.numeric(pubsplit[i,]), as.numeric(locsplit[i,])), nrow=2, byrow=TRUE), k = 5, verbose=FALSE, seed=450+i)$top.list, collapse=" ")
#  }
# stopCluster(cl)

w=matrix(rep(c(5,4,3,2,1),5), ncol=5, byrow=TRUE)
imp <- c(4,rep(1,4))

# cl <- makeCluster(14)
# registerDoParallel(cl)
# matlist <- foreach(i=1:nrow(bestpub), .combine=list, .packages=c("data.table")) %dopar% {
#   matrix(as.character(c(pub_split[i,], xgb16_split[i,], xgb17_split[i,], xgb18_split[i,], xgb19_split[i,], xgb20_split[i,], xgb21_split[i,], xgb1_loop_split[i,], xgb2_loop_split[i,], xgb3_loop_split[i,], xgb5_loop_split[i,])), ncol=5, byrow=TRUE)
# }
# stopCluster(cl)

cl <- makeCluster(13)
registerDoParallel(cl)
raVec <- foreach(i=1:nrow(bestpub), .combine=c, .packages=c("data.table", "RankAggreg")) %dopar% {
  paste(BruteAggreg(matrix(as.character(c(pub_split[i,], xgb1_loop_split[i,], xgb2_loop_split[i,], xgb3_loop_split[i,], xgb5_loop_split[i,])), ncol=5, byrow=TRUE), k = 5, weights=w, distance="Spearman", importance=imp)$top.list, collapse=" ")
}
stopCluster(cl)

submission <- data.table(id=seq(0,nrow(bestpub)-1,1), hotel_cluster=raVec)
fwrite(submission, "./stack_models/L1/testPreds/L1_testPreds_RankAggreg_v4.csv")

