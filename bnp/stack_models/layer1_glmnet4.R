library(caret)
library(glmnet)
library(readr)
library(data.table)
library(doParallel)
library(gtools)
setwd("/media/branden/SSHD1/kaggle/bnp")
threads <- ifelse(detectCores()>8,detectCores()-8,detectCores()-4)
ts1Trans <- fread("./data_trans/ts2Trans_v20.csv")
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


pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","center","scale"), thresh=0.999)
# pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","center","scale"))
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
glmnetGrid <- expand.grid(alpha=c(.003,.01,.03), lambda=c(.01,.03,0.1))

cl <- makeCluster(6)
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
glmnet4 <- train(x=train_pca,
                 y=as.factor(make.names(ts1Trans[ts1Trans$filter==0,target])),
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss")
stopCluster(cl)
Sys.time() - tme
save(glmnet4, file="./stack_models/layer1_glmnet4.rda")

cvPreds <- glmnet4$pred[glmnet4$pred$alpha==glmnet4$bestTune$alpha & glmnet4$pred$lambda==glmnet4$bestTune$lambda,c(3,5)]
cvPreds <- cvPreds[order(cvPreds$rowIndex),]
cvPreds$rowIndex <- NULL

colnames(cvPreds) <- "PredictedProb_glmnet4"
write_csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], cvPreds), "./stack_models/cvPreds/cvPreds_glmnet4.csv") 

# Test Predictions and Submission file
preds <- predict(glmnet4, newdata=test_pca, type="prob")[,2]
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=samp$ID, TARGET=preds)
write_csv(submission, "./stack_models/testPreds/testPreds_glmnet4.csv")

