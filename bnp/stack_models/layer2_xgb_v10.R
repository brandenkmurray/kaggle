library(readr)
library(data.table)
library(xgboost)
library(reshape2)
library(dplyr)
library(caret)
setwd("/media/branden/SSHD1/kaggle/bnp")
load("./data_trans/cvFoldsList_lay2.rda")

#################################
## Load CV predictions from models
#################################
temp = list.files(path="./stack_models/cvPreds/", pattern="*.csv")

temp <- c("cvPreds_glmnet3.csv","cvPreds_nn3.csv" ,"cvPreds_et1.csv" ,"cvPreds_et2.csv" ,"cvPreds_xgb17.csv","cvPreds_xgb18.csv","cvPreds_xgb20.csv","cvPreds_xgb21.csv","cvPreds_xgb22.csv","cvPreds_xgb24.csv","cvPreds_xgb25.csv","cvPreds_xgb26.csv","cvPreds_xgb27.csv","cvPreds_xgb28.csv","cvPreds_xgb29.csv")

cvPredsList <- list()
for (i in grep(pattern="cvPreds",temp)) {
  cvPredsList[[paste0(temp[i])]] <- assign(temp[i], read.csv(paste0("./stack_models/cvPreds/",temp[i])))
}

# Edit and bind predictions
cvPredsList <- lapply(cvPredsList, function(x) {colnames(x)[1] <- c("ID"); x})
cvPredsList <- lapply(cvPredsList, function(x) x[order(x$ID),])
cvPredsList <- lapply(cvPredsList, function(x) {x["ID"] <- NULL; x})


lay1preds <- as.data.frame(cvPredsList)



# #################
# ## Load Rafa's CV predictions
# ################
rafaPreds <- read_csv("/media/branden/SSHD1/kaggle/bnp/Rafa/branden_exp/probs_stage_1/train.csv")
colnames(rafaPreds)[1] <- "ID"
rafaPreds <- rafaPreds[order(rafaPreds$ID),]

lay1preds <- cbind(lay1preds, rafaPreds[,2:ncol(rafaPreds)])

# Add the class column to the dataset
t1 <- data.table(read.csv("train.csv"))
target <- t1$target[order(t1$ID)]

lay1preds <- cbind(target=target, lay1preds)

########################
## Load Test Set predictions from models trained on the entire training set
########################
temp = list.files(path="./stack_models/testPreds/", pattern="*.csv")
temp <- c("testPreds_glmnet3.csv","testPreds_nn3.csv","testPreds_et1.csv","testPreds_et2.csv","testPreds_xgb17.csv","testPreds_xgb18.csv","testPreds_xgb20.csv","testPreds_xgb21.csv","testPreds_xgb22.csv","testPreds_xgb24.csv","testPreds_xgb25.csv","testPreds_xgb26.csv","testPreds_xgb27.csv","testPreds_xgb28.csv","testPreds_xgb29.csv")

testPredsList <- list()
for (i in grep(pattern="testPreds",temp)) {
  testPredsList[[paste0(temp[i])]] <- assign(temp[i], read.csv(paste0("./stack_models/testPreds/",temp[i])))
}

# Edit and bind predictions
testPredsList <- lapply(testPredsList, function(x) {colnames(x)[1] <- c("ID"); x})
testPredsList <- lapply(testPredsList, function(x) x[order(x$ID),])
testPredsList <- lapply(testPredsList, function(x) {x["ID"] <- NULL; x})

lay1fullpreds <- as.data.frame(testPredsList)


# #################
# ## Load Rafa's Test predictions
# ################
rafaTestPreds <- read_csv("/media/branden/SSHD1/kaggle/bnp/Rafa/branden_exp/probs_stage_1/test.csv")
colnames(rafaTestPreds)[1] <- "ID"
rafaTestPreds <- rafaTestPreds[order(rafaTestPreds$ID),]

lay1fullpreds <- cbind(lay1fullpreds, rafaTestPreds[,2:ncol(rafaTestPreds)])


####################
## Train ensemble
####################

dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds)]),label=data.matrix(lay1preds[,"target"]))

# Train Model
# eta=0.01, md=7, mcw=1, ss=0.75, csbt=0.75, xgb1-6 = 0.447017+0.005724

param <- list(objective="binary:logistic",
              eval_metric="logloss",
              eta = .01,
              max_depth=6,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              nthread=12
)

#Train CV models
set.seed(201601)
(tme <- Sys.time())
xgbLay2_v10 <- xgb.cv(data = dtrain,
                         params = param,
                         nrounds = 50000,
                         maximize=FALSE,
                         prediction=TRUE,
                         folds=cvFoldsList_lay2,
                         # watchlist=watchlist,
                         print.every.n = 50,
                         early.stop.round=200)
Sys.time() - tme
save(xgbLay2_v10, file="./stack_models/xgbLay2_v10.rda")
cvPreds <- data.frame(ID=t1$ID, PredictProb=xgbLay2_v10$pred)
write_csv(cvPreds, "./stack_models/layer2Preds/cvPreds_xgbLay2_v10.csv")
rounds <- floor(which.min(xgbLay2_v10$dt$test.logloss.mean) * 1.08)

# Train on all observations
set.seed(201601)
(tme <- Sys.time())
xgbLay2_v10_full <- xgb.train(data = dtrain,
                        params = param,
                        nrounds = rounds,
                        maximize=FALSE,
                        # watchlist=watchlist,
                        print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v10_full, file="./stack_models/xgbLay2_v10_full.rda")

lay2preds <- predict(xgbLay2_v10_full, newdata=data.matrix(lay1fullpreds))
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=sort(samp$ID), PredictedProb=lay2preds)
write.csv(submission, "./stack_models/layer2Preds/xgbLay2_v10preds.csv", row.names=FALSE)

xgbLay2_v10_Imp <- xgb.importance(feature_names = colnames(lay1preds[,2:ncol(lay1preds)]), model=xgbLay2_v10_full)
write_csv(xgbLay2_v10_Imp, "./stack_models/xgbLay2_v10_Imp.csv")








library(caret)
library(doParallel)
load("./data_trans/cvFoldsTrainList_lay2.rda")

lay1preds_trans <- lay1preds[,2:ncol(lay1preds)]
lay1preds_trans[lay1preds_trans==0] <- 1e-15
lay1preds_trans[lay1preds_trans==1] <- 1-1e-15
lay1preds_trans <- log((lay1preds_trans/(1-lay1preds_trans)))

lay1fullpreds_trans <- lay1fullpreds
lay1fullpreds_trans[lay1fullpreds_trans==0] <- 1e-15
lay1fullpreds_trans[lay1fullpreds_trans==1] <- 1-1e-15
lay1fullpreds_trans <- log((lay1fullpreds_trans/(1-lay1fullpreds_trans)))

glmnetControl <- trainControl(method="cv",
                              number=6,
                              summaryFunction=mnLogLoss,
                              savePredictions=TRUE,
                              classProbs=TRUE,
                              index=cvFoldsTrainList_lay2,
                              allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.001,.003,.01,.03,.1), lambda=c(.0001,.0003,.001,.003,.01,.03))

cl <- makeCluster(6)
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
lay2_glmnet1 <- train(x=lay1preds_trans,
                 y=as.factor(make.names(lay1preds$target)),
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss")
stopCluster(cl)
Sys.time() - tme

lay2_glmnet1
cvPredsGLM <- lay2_glmnet1$pred[lay2_glmnet1$pred$alpha==lay2_glmnet1$bestTune$alpha & lay2_glmnet1$pred$lambda==lay2_glmnet1$bestTune$lambda,c(3,5)]
cvPredsGLM <- cvPredsGLM[order(cvPredsGLM$rowIndex),]
cvPreds$rowIndex <- NULL

testPredsGLM <- predict(lay2_glmnet1, lay1fullpreds_trans, type="prob")[,2]

LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}

blend <- cvPreds$PredictProb*.7 + cvPredsGLM$X1*.3
LogLoss(lay1preds$target, blend)

testBlend <- lay2preds*.7  + testPredsGLM*.3
head(testBlend)
submission <- data.frame(ID=sort(samp$ID), PredictedProb=testBlend)
write.csv(submission, "./stack_models/layer2Preds/xgbLay2_v10testBlend.csv", row.names=FALSE)
