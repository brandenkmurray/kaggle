library(readr)
library(data.table)
library(caret)
library(reshape2)
library(dplyr)
library(kknn)
library(Matrix)
library(doParallel)
library(Metrics)
setwd("/media/branden/SSHD1/kaggle/bnp")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)
ts1Trans <- data.table(read.csv("./data_trans/ts2Trans_v10.csv"))
load("./data_trans/cvFoldsList.rda")


# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}


# Only do KNN with summary variables and Departments
kknnPre <- preProcess(ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE], method=c("zv","pca", "center", "scale"), pcaComp=400)
t1kknn <- predict(kknnPre, ts1Trans[filter==0,4:ncol(ts1Trans), with=FALSE])
s1kknn <- predict(kknnPre, ts1Trans[filter==2,4:ncol(ts1Trans), with=FALSE])

t1kknn$target <- ts1Trans[filter==0, "target",with=FALSE]
t1kknn$target <- as.factor(make.names(t1kknn$target))

kknn_k <- 50
kknn_dist <- 1
kknn_kernel <- "epanechnikov"

# Parallelize

# kknn1_stack_preds <- matrix(0, nrow=nrow(t1kknn), ncol=1)
# logLossTable <- data.frame(fold=seq(1:length(cvFoldsList)), LogLoss=rep(0, length(cvFoldsList)))
# for (i in 1:length(cvFoldsList)){
#   kknn1 <- kknn(target ~ .,
#                 train=t1kknn[-cvFoldsList[[1]],], 
#                 test=t1kknn[cvFoldsList[[1]],], 
#                 k=kknn_k, 
#                 distance=kknn_dist,
#                 kernel=kknn_kernel )
#   kknn1_stack_preds[cvFoldsList[[1]],] <- kknn1$prob[,2]
#   logLossTable[i,2] <- LogLoss(as.numeric(t1kknn$target[cvFoldsList[[1]]])-1, kknn1$prob[,2])
# }

(tme <- Sys.time())
cl <- makeCluster(5)
registerDoParallel(cl)
out <- foreach(i=1:length(cvFoldsList), .combine="c", .packages=c("kknn")) %dopar% {
  kknn1 <- kknn(target ~ .,
                train=t1kknn[-cvFoldsList[[i]],], 
                test=t1kknn[cvFoldsList[[i]],], 
                k=kknn_k, 
                distance=kknn_dist,
                kernel=kknn_kernel )
  kknn1$prob[,2]
}
(Sys.time() - tme)
stopCluster(cl)

kknn1_stack_preds <- data.frame(index=unlist(cvFoldsList), cvPred=out)
kknn1_stack_preds <- kknn1_stack_preds[order(kknn1_stack_preds$index),]
LogLoss(as.numeric(t1kknn$target)-1, kknn1_stack_preds$cvPred)


cvPreds <- kknn1_stack_preds$cvPred
samp <- read.csv('sample_submission.csv')
cnames <- paste("kknn1", names(samp)[2:ncol(samp)], sep="_")
names(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"ID",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds/cvPreds_kknn1.csv", row.names=FALSE)

# LogLoss(model.matrix(~class-1, t1kknn[-cvFoldsList[[5]],"class", with=FALSE]), kknn1_stack_preds$prob)
# LogLoss(model.matrix(~class-1, t1kknn[,"class", with=FALSE]), kknn1_stack_preds)


kknn1full <- kknn(target ~ .,
              train=t1kknn, 
              test=s1kknn, 
              k=kknn_k, 
              distance=kknn_dist,
              kernel=kknn_kernel)
save(kknn1full, file="./stack_models/kknn1full.rda")

testPreds <- kknn1full$prob[,2]
names(testPreds) <- "PredictedProb"
testPreds <- cbind(ts1Trans[filter==2,"ID",with=FALSE], testPreds)
write.csv(testPreds, "./stack_models/testPreds/testPreds_kknn1.csv", row.names=FALSE)
