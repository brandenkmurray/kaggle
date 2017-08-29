library(data.table)
library(RankAggreg)
library(doParallel)
library(doRNG)
bestpub <- fread("./stack_models/L1/testPreds/best_public_submission_2016-05-30-15-44.csv", header=TRUE)
bestloc <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb16-loop5_wtdavg.csv", header=TRUE)

pubsplit <- data.frame(do.call('rbind', strsplit(as.character(bestpub$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)
locsplit <- data.frame(do.call('rbind', strsplit(as.character(bestloc$hotel_cluster),' ',fixed=TRUE)), stringsAsFactors = FALSE)


x <- do.call("paste", c(c(pubsplit[1,], locsplit[1,]), sep=" "))


cl <- makeCluster(14)
registerDoParallel(cl)
raVec <- foreach(i=1:nrow(bestloc), .combine="c", .packages=c("data.table", "RankAggreg")) %dopar% {
  paste(RankAggreg(matrix(c(as.numeric(pubsplit[i,]), as.numeric(locsplit[i,])), nrow=2, byrow=TRUE), k = 5, verbose=FALSE, seed=450+i)$top.list, collapse=" ")
}
stopCluster(cl)

submission <- data.table(id=seq(0,nrow(bestloc)-1,1), hotel_cluster=raVec)
fwrite(submission, "./stack_models/L1/testPreds/L1_testPreds_RankAggreg_v1.csv")



