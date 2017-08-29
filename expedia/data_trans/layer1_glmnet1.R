library(data.table)
library(Matrix)
library(gtools)
library(Metrics)
library(xgboost)
library(doParallel)
# setwd("/media/branden/SSHD1/kaggle/expedia")
# setwd("/media/branden/SSHD1/kaggle/expedia")
setwd("~/ebs")
# load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
map5 <- function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = t(matrix(preds, ncol = length(labels)))
  preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
  return (list(metric = 'map5', value = map5))
}
#######################
## Load data
#######################
ts1 <-  fread("./data_trans/ts1_merged_v1.csv", key="id2")
userpref <- fread("./data_trans/data_trans_userpref_top5.csv", key = "id2")

ts1 <- ts1[userpref]

varnames <- colnames(ts1)[!colnames(ts1) %in% c("cnt","id","id2","hotel_cluster","filter","dummy","pred0","date_time", "srch_ci","srch_co","date")]


# # Was only necessary for easier filtering of the validation set
# train <- ts1Trans[ts1Trans$filter==0,]
# test <- ts1Trans[ts1Trans$filter==2,]
# # Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
# train$target <- as.factor(make.names(train$target))
# # 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)


# pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), thresh=0.999)
pca <- preProcess(ts1Trans[filter==0,varnames, with=FALSE], method=c("zv","BoxCox","center","scale"))
# train_pca <- predict(pca, train[,4:ncol(train), with=FALSE])
# test_pca <- predict(pca, newdata=test[,4:ncol(test), with=FALSE])
ts1Trans <- predict(pca, newdata=ts1Trans[,varnames,with=FALSE])

# rm(train, test)

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
glmnet5 <- train(x=train_pca,
                 y=as.factor(make.names(ts1Trans[ts1Trans$filter==0,target])),
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss")
stopCluster(cl)
Sys.time() - tme
save(glmnet5, file="./stack_models/layer1_glmnet5.rda")

cvPreds <- glmnet5$pred[glmnet5$pred$alpha==glmnet5$bestTune$alpha & glmnet5$pred$lambda==glmnet5$bestTune$lambda,c(3,5)]
cvPreds <- cvPreds[order(cvPreds$rowIndex),]
cvPreds$rowIndex <- NULL

colnames(cvPreds) <- "PredictedProb_glmnet5"
write_csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], cvPreds), "./stack_models/cvPreds/cvPreds_glmnet5.csv") 

# Test Predictions and Submission file
preds <- predict(glmnet5, newdata=test_pca, type="prob")[,2]
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=samp$ID, TARGET=preds)
write_csv(submission, "./stack_models/testPreds/testPreds_glmnet5.csv")

