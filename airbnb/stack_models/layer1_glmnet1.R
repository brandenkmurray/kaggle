library(caret)
library(glmnet)
library(readr)
library(data.table)
library(doParallel)
library(gtools)
setwd("/home/branden/Documents/kaggle/airbnb")
threads <- ifelse(detectCores()>8,detectCores()-8,detectCores()-2)
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v4.csv"))
xgb6cv <- read_csv("./stack_models/cvPreds_xgb6.csv")
# varnames <- names(which(sapply(ts1Trans[filter==0, 5:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))


# Was only necessary for easier filtering of the validation set
train <- ts1Trans[ts1Trans$filter==0,]
test <- ts1Trans[ts1Trans$filter==2,]
# Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
train$class <- as.factor(make.names(train$class))
# 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)


pca <- preProcess(train[,4:ncol(train), with=FALSE], method=c("zv","BoxCox","pca","center","scale"), thresh=0.99)
train_pca <- predict(pca, train[,4:ncol(train), with=FALSE])
test_pca <- predict(pca, newdata=test[,4:ncol(test), with=FALSE])


# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}


glmnetControl <- trainControl(method="cv",
                              number=5,
                              summaryFunction=mnLogLoss,
                              savePredictions=TRUE,
                              classProbs=TRUE,
                              index=cvFoldsTrainList,
                              allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.003,.01,.003,.1,.3), lambda=c(.003,.01,.03))

cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
glmnet1 <- train(x=train_pca,
                 y=train$class,
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss")
stopCluster(cl)
Sys.time() - tme
save(glmnet1, file="./stack_models/layer1_glmnet1.rda")

cvPreds <- glmnet1$pred[glmnet1$pred$alpha==glmnet1$bestTune$alpha & glmnet1$pred$lambda==glmnet1$bestTune$lambda,3:15]
cvPreds <- cvPreds[order(cvPreds$rowIndex),mixedorder(names(cvPreds))]
cvPreds$rowIndex <- NULL

samp <- read_csv('sample_submission_NDF.csv')
cnames <- paste("glmnet1", names(cvPreds)[1:ncol(cvPreds)], sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1Trans[filter==0,"id",with=FALSE], cvPreds), "./stack_models/cvPreds_glmnet1.csv", row.names=FALSE) 


preds <- predict(glmnet1, newdata=test_pca, type="prob")
preds <- preds[,mixedorder(names(preds))]
samp <- read_csv('sample_submission_NDF.csv')
classMap <- read_csv("./data_trans/classMap.csv")
colnames(preds) <- paste("glmnet1",classMap$country_destination,sep="_")
sampID <- read_csv('sample_submission_NDF.csv')$id
sampID <- sort(sampID)
write.csv(data.frame(id=sampID, preds), "./stack_models/testPredsProbs_glmnet1.csv", row.names=FALSE)
colnames(preds) <- classMap$country_destination
testPreds_top5 <- as.vector(apply(preds, 1, function(x) names(sort(x)[12:8])))



# cnames <- names(samp)[2:ncol(samp)]
# names(preds) <- cnames
# submission <- data.frame(id=sort(samp$id), preds)
# write.csv(submission, "./stack_models/testPreds_glmnet1.csv", row.names=FALSE)
# 
# 
# lay2Preds <- predict(xgbLay2_v3_full, newdata=data.matrix(lay1fullpreds))
# lay2Preds <- as.data.frame(matrix(lay2Preds, nrow=12))
# classMap <- read_csv("./data_trans/classMap.csv")
# rownames(lay2Preds) <- classMap$country_destination
# sampID <- read_csv('sample_submission_NDF.csv')$id
# sampID <- sort(sampID)
# write.csv(data.frame(id=sampID, t(lay2Preds)), "./stack_models/lay2PredsProbs_xgb_v3.csv", row.names=FALSE)
# testPreds_top5 <- as.vector(apply(lay2Preds, 2, function(x) names(sort(x)[12:8])))


# create submission 
idx = sampID
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/testPreds_glmnet1.csv", quote=FALSE, row.names = FALSE)

