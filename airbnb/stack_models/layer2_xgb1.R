library(data.table)
library(readr)
library(xgboost)
setwd("/home/branden/Documents/kaggle/airbnb")
load("./data_trans/cvFoldsList_lay2.rda")

xgb1cv <- read_csv("./stack_models/cvPreds_xgb1.csv") 
xgb2cv <- read_csv("./stack_models/cvPreds_xgb2.csv") 
xgb3cv <- read_csv("./stack_models/cvPreds_xgb3.csv") 
xgb4cv <- read_csv("./stack_models/cvPreds_xgb4.csv") 
xgb5cv <- read_csv("./stack_models/cvPreds_xgb5.csv") 
xgb6cv <- read_csv("./stack_models/cvPreds_xgb6.csv") 

xgb1cv$id <- NULL
xgb2cv$id <- NULL
xgb3cv$id <- NULL
xgb4cv$id <- NULL
xgb5cv$id <- NULL
xgb6cv$id <- NULL

lay1preds <- cbind(xgb1cv, xgb2cv, xgb3cv, xgb4cv, xgb5cv, xgb6cv)

# Get actual classes
t1 <- data.table(read.csv("./train_users_2.csv"))
classMap <- read_csv("./data_trans/classMap.csv")
t1 <- merge(t1, classMap, by="country_destination")
t1 <- t1[order(t1$id),]
country_destination<- t1$country_destination

lay1preds <- data.table(cbind(class=t1$class, lay1preds))

dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[,"class", with=FALSE]))


# Train Model
# eta=0.01, md=3, mcw=1,ss=0.7,csbt=0.7, xgb1, xgb2, xgb3, xgb4, xgb5, xgb6 = 1.022082          0.004287

param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=12,
              eta = .01,
              max_depth=3,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7
)
set.seed(201510)
(tme <- Sys.time())
xgbLay2_v1 <- xgb.cv(data = dtrain,
                      params = param,
                      nrounds = 6000,
                      maximize=FALSE,
                      prediction=TRUE,
                      folds=cvFoldsList_lay2,
                      # watchlist=watchlist,
                      print.every.n = 5,
                      early.stop.round=50)
Sys.time() - tme
save(xgbLay2_v1, file="./stack_models/xgbLay2_v1.rda")

xgbLay2_v1$dt[which.min(xgbLay2_v1$dt$test.mlogloss.mean),]
rounds <- floor(which.min(xgbLay2_v1$dt$test.mlogloss.mean) * 1.15)

# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read_csv("./stack_models/testPredsProbs_xgb1.csv")
xgb2fullpreds <- read_csv("./stack_models/testPredsProbs_xgb2.csv")
xgb3fullpreds <- read_csv("./stack_models/testPredsProbs_xgb3.csv")
xgb4fullpreds <- read_csv("./stack_models/testPredsProbs_xgb4.csv")
xgb5fullpreds <- read_csv("./stack_models/testPredsProbs_xgb5.csv")
xgb6fullpreds <- read_csv("./stack_models/testPredsProbs_xgb6.csv")
# Edit and bind test set predictions
xgb1fullpreds$id <- NULL
xgb2fullpreds$id <- NULL
xgb3fullpreds$id <- NULL
xgb4fullpreds$id <- NULL
xgb5fullpreds$id <- NULL
xgb6fullpreds$id <- NULL

lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds, xgb3fullpreds, xgb4fullpreds, xgb5fullpreds, xgb6fullpreds)
# Predict the test set using the XGBOOST stacked model

set.seed(201510)
(tme <- Sys.time())
xgbLay2_v1_full <- xgb.train(data = dtrain,
                              params = param,
                              nrounds = rounds,
                              maximize=FALSE,
                              # watchlist=watchlist,
                              print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v1_full, file="./stack_models/xgbLay2_v1_full.rda")

lay2Preds <- predict(xgbLay2_v1_full, newdata=data.matrix(lay1fullpreds))
lay2Preds <- as.data.frame(matrix(lay2Preds, nrow=12))
classMap <- read_csv("./data_trans/classMap.csv")
rownames(lay2Preds) <- classMap$country_destination
sampID <- read_csv('sample_submission_NDF.csv')$id
sampID <- sort(sampID)
write.csv(data.frame(id=sampID, t(lay2Preds)), "./stack_models/lay2PredsProbs_xgb1.csv", row.names=FALSE)
testPreds_top5 <- as.vector(apply(lay2Preds, 2, function(x) names(sort(x)[12:8])))


# create submission 
idx = sampID
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/lay2Preds_xgb1.csv", quote=FALSE, row.names = FALSE)


