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

cvPredsList <- list()
for (i in grep(pattern="cvPreds",temp)) {
  cvPredsList[[paste0(temp[i])]] <- assign(temp[i], read.csv(paste0("./stack_models/cvPreds/",temp[i])))
}

# Edit and bind predictions
cvPredsList <- lapply(cvPredsList, function(x) {colnames(x)[1] <- c("ID"); x})
cvPredsList <- lapply(cvPredsList, function(x) x[order(x$ID),])
cvPredsList <- lapply(cvPredsList, function(x) {x["ID"] <- NULL; x})

lay1preds <- as.data.frame(cvPredsList)

# Add the class column to the dataset
t1 <- data.table(read.csv("train.csv"))
target <- t1$target[order(t1$ID)]

lay1preds <- cbind(target=target, lay1preds)

########################
## Load Test Set predictions from models trained on the entire training set
########################
temp = list.files(path="./stack_models/testPreds/", pattern="*.csv")

testPredsList <- list()
for (i in grep(pattern="testPreds",temp)) {
  testPredsList[[paste0(temp[i])]] <- assign(temp[i], read.csv(paste0("./stack_models/testPreds/",temp[i])))
}

# Edit and bind predictions
testPredsList <- lapply(testPredsList, function(x) {colnames(x)[1] <- c("ID"); x})
testPredsList <- lapply(testPredsList, function(x) x[order(x$ID),])
testPredsList <- lapply(testPredsList, function(x) {x["ID"] <- NULL; x})

lay1fullpreds <- as.data.frame(testPredsList)


####################
## Train ensemble
####################

dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds)]),label=data.matrix(lay1preds[,"target"]))

# Train Model
# eta=0.01, md=7, mcw=1, ss=0.75, csbt=0.75, xgb1-6 = 0.447017+0.005724

param <- list(objective="binary:logistic",
              eval_metric="logloss",
              eta = .01,
              max_depth=7,
              min_child_weight=1,
              subsample=.65,
              colsample_bytree=.65,
              nthread=12
)

#Train CV models
set.seed(201601)
(tme <- Sys.time())
xgbLay2_v4 <- xgb.cv(data = dtrain,
                         params = param,
                         nrounds = 50000,
                         maximize=FALSE,
                         prediction=TRUE,
                         folds=cvFoldsList_lay2,
                         # watchlist=watchlist,
                         print.every.n = 50,
                         early.stop.round=200)
Sys.time() - tme
save(xgbLay2_v4, file="./stack_models/xgbLay2_v4.rda")
cvPreds <- data.frame(ID=t1$ID, PredictProb=xgbLay2_v4$pred)
write_csv(cvPreds, "./stack_models/layer2Preds/cvPreds_xgbLay2_v4.csv")
rounds <- floor(which.min(xgbLay2_v4$dt$test.logloss.mean) * 1.08)

# Train on all observations
set.seed(201601)
(tme <- Sys.time())
xgbLay2_v4_full <- xgb.train(data = dtrain,
                        params = param,
                        nrounds = rounds,
                        maximize=FALSE,
                        # watchlist=watchlist,
                        print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v4_full, file="./stack_models/xgbLay2_v4_full.rda")

lay2preds <- predict(xgbLay2_v4_full, newdata=data.matrix(lay1fullpreds))
samp <- read_csv('sample_submission.csv')
submission <- data.frame(ID=sort(samp$ID), PredictedProb=lay2preds)
write.csv(submission, "./stack_models/layer2Preds/xgbLay2_v4preds.csv", row.names=FALSE)

xgbLay2_v4_Imp <- xgb.importance(feature_names = colnames(lay1preds[,2:ncol(lay1preds)]), model=xgbLay2_v4_full)
write_csv(xgbLay2_v4_Imp, "./stack_models/xgbLay2_v4_Imp.csv")
