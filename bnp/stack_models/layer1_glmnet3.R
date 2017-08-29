library(caret)
library(glmnet)
library(readr)
library(data.table)
library(doParallel)
library(gtools)
setwd("/media/branden/SSHD1/kaggle/bnp")
threads <- ifelse(detectCores()>8,detectCores()-8,detectCores()-4)
ts1Trans <- data.table(read.csv("./data_trans/ts2Trans_v16.csv"))
load("./data_trans/cvFoldsTrainList.rda")
# varnames <- names(which(sapply(ts1Trans[filter==0, 5:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))


# Was only necessary for easier filtering of the validation set
train <- ts1Trans[ts1Trans$filter==0,]
test <- ts1Trans[ts1Trans$filter==2,]
# Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
train$target <- as.factor(make.names(train$target))
# 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)


# pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), thresh=0.999)
pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","center","scale"))
train_pca <- predict(pca, train[,4:ncol(train), with=FALSE])
test_pca <- predict(pca, newdata=test[,4:ncol(test), with=FALSE])

rm(train, test)
# Logloss function
# LogLoss <- function(actual, predicted, eps=1e-15) {
#   predicted[predicted < eps] <- eps;
#   predicted[predicted > 1 - eps] <- 1 - eps;
#   -1/nrow(actual)*(sum(actual*log(predicted)))
# }


glmnetControl <- trainControl(method="cv",
                              number=5,
                              summaryFunction=mnLogLoss,
                              savePredictions=TRUE,
                              classProbs=TRUE,
                              index=cvFoldsTrainList,
                              allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.003,.01), lambda=c(.01,.03))

cl <- makeCluster(4)
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
glmnet3 <- train(x=train_pca,
                 y=as.factor(make.names(ts1Trans[ts1Trans$filter==0,target])),
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss")
stopCluster(cl)
Sys.time() - tme
save(glmnet3, file="./stack_models/layer1_glmnet3.rda")

cvPreds <- glmnet3$pred[glmnet3$pred$alpha==glmnet3$bestTune$alpha & glmnet3$pred$lambda==glmnet3$bestTune$lambda,c(3,5)]
cvPreds <- cvPreds[order(cvPreds$rowIndex),]
cvPreds$rowIndex <- NULL

colnames(cvPreds) <- "PredictedProb_glmnet3"
write_csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], cvPreds), "./stack_models/cvPreds/cvPreds_glmnet3.csv") 

# Test Predictions and Submission file
preds <- predict(glmnet3, newdata=test_pca, type="prob")[,2]
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=samp$ID, TARGET=preds)
write_csv(submission, "./stack_models/testPreds/testPreds_glmnet3.csv")

