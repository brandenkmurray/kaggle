library(caret)
library(RSofia)
library(readr)
library(data.table)
library(doParallel)
library(gtools)
setwd("/media/branden/SSHD1/kaggle/bnp")
threads <- ifelse(detectCores()>8,detectCores()-8,detectCores()-2)

#################
# Logloss function
#################
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}
#################

ts1Trans <- data.table(read.csv("./data_trans/ts2Trans_v11.csv"))
load("./data_trans/cvFoldsList.rda")
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))


# Was only necessary for easier filtering of the validation set
train <- ts1Trans[ts1Trans$filter==0,]
test <- ts1Trans[ts1Trans$filter==2,]
# Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
# train$target <- as.factor(make.names(train$target))
# 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)


pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), pcaComp =300)
train_pca <- predict(pca, train[,4:ncol(train), with=FALSE])
test_pca <- predict(pca, newdata=test[,4:ncol(test), with=FALSE])
train_pca$target  <- ts1Trans[filter==0, target]
test_pca$target  <- -1

# Train model
(tme <- Sys.time())
cl <- makeCluster(5)
registerDoParallel(cl)
out <- foreach(i=1:length(cvFoldsList), .combine="c", .packages=c("RSofia")) %dopar% {
  sofia1 <- sofia(target ~ .,
                  # data=train[-cvFoldsList[[i]], c("target", varnames),with=FALSE],
                data=train_pca[-cvFoldsList[[i]],], 
                random_seed=400,
                lambda=0.2,
                iterations=1e+06,
                learner_type="logreg-pegasos", 
                eta_type="pegasos",
                loop_type="stochastic")
  predict(sofia1, train_pca[cvFoldsList[[i]],], prediction_type="logistic")
}
(Sys.time() - tme)
stopCluster(cl)

cvPreds <- data.frame(index=unlist(cvFoldsList), PredictedProb_sofia1=out)
cvPreds <- cvPreds[order(cvPreds$index),]
LogLoss(train_pca$target, cvPreds$PredictedProb_sofia1)

write_csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], cvPreds[,2]), "./stack_models/cvPreds/cvPreds_sofia1.csv") 

# Train on full dataset
sofia1 <- sofia(target ~ .,
                # data=train[-cvFoldsList[[i]], c("target", varnames),with=FALSE],
                data=train_pca, 
                random_seed=400,
                lambda=0.2,
                iterations=1e+06,
                learner_type="logreg-pegasos", 
                eta_type="pegasos",
                loop_type="stochastic")


# Test Predictions and Submission file
preds <- predict(sofia1, test_pca, prediction_type="logistic")
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=samp$ID, PredictedProb=preds)
write_csv(submission, "./stack_models/testPreds/testPreds_sofia1.csv")

