library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
setwd("/home/branden/Documents/kaggle/walmart")

# Load CV predictions from models
load("./stack_models/xgbLay2_v14.rda")
load("./stack_models/xgbLay2_v15.rda")
nnLay2_v1 <- read_csv("./stack_models/cvPreds_lay2_nn1.csv")

nnLay2_v1$VisitNumber <- NULL

lay2preds <- cbind(xgbLay2_v15$pred, nnLay2_v1)
# lay1preds <- cbind(xgb1preds,xgb7preds, kknn1preds, nn1preds, nn2preds)
# Add the class column to the dataset
t1 <- data.table(read_csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class)]

lay2preds <- data.table(cbind(class=t1$class, lay2preds))

# Create a validation set
# set.seed(1234)
# h <- sample(nrow(lay1preds), 2000)
# # Create DMatrices
# dval <- xgb.DMatrix(data=data.matrix(lay1preds[h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[h,"class", with=FALSE]))
# dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[-h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[-h,"class", with=FALSE]))
# watchlist <- list(val=dval,train=dtrain)

load("./data_trans/cvFoldsList_lay2.rda")
set.seed(76)
cvFoldsList_lay3 <- createFolds(lay2preds$class, k=6, list=TRUE, returnTrain=FALSE)
dtrain <- xgb.DMatrix(data=data.matrix(lay2preds[,2:ncol(lay2preds), with=FALSE]),label=data.matrix(lay2preds[,"class", with=FALSE]))



# Train Model
# xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.564680+0.003101
# xgb1, xgb2, xgb3, kknn1, nn1 = 0.567281+0.002696
# xgb1, xgb7, kknn1, nn1, nn2 = 0.562542+0.003631
# subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.559679+0.002955
# eta=0.02, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2, et1 = 0.556184+0.002734
# eta=0.02,maxdep=3, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, xgb8,xgb10,kknn1, nn1, nn2, et1,glmnet1 = 0.54421+0.003521
# eta=0.02,maxdep=3, minchild=1, subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, xgb8,xgb10,xgb11,kknn1, kknn2,kknn3,kknn4,nn1, nn2, et1,glmnet1,glmnet2 = 
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .03,
              max_depth=3,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              nthread=12
)
set.seed(201510)
(tme <- Sys.time())
xgbLay3_v1 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               prediction=TRUE,
               folds=cvFoldsList_lay3,
               # watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgbLay3_v1, file="./stack_models/xgbLay3_v1.rda")

rounds <- floor(which.min(xgbLay3_v1$dt$test.mlogloss.mean) * 1.15)

# Load Test Set predictions from models trained on the entire training set
xgbLay2_v14_fullpreds <- read_csv("./stack_models/xgbLay2_v14preds.csv")
xgbLay2_v15_fullpreds <- read_csv("./stack_models/xgbLay2_v15preds.csv")
nnLay2_v1_fullpreds <- read_csv("./stack_models/nnlay2_v1preds.csv")

# Edit and bind test set predictions
xgbLay2_v14_fullpreds$VisitNumber <- NULL
xgbLay2_v15_fullpreds$VisitNumber <- NULL
nnLay2_v1_fullpreds$VisitNumber <- NULL
lay2fullpreds <- cbind(xgbLay2_v15_fullpreds, nnLay2_v1_fullpreds)
# lay1fullpreds <- cbind(xgb1fullpreds,xgb7fullpreds, kknn1fullpreds, nn1fullpreds, nn2fullpreds)
# Predict the test set using the XGBOOST stacked model

set.seed(201510)
(tme <- Sys.time())
xgbLay2_v16_full <- xgb.train(data = dtrain,
                        params = param,
                        nrounds = rounds,
                        maximize=FALSE,
                        # watchlist=watchlist,
                        print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v16_full, file="./stack_models/xgbLay2_v16_full.rda")

lay2preds <- predict(xgbLay2_v16_full, newdata=data.matrix(lay1fullpreds))
preds <- data.frame(t(matrix(lay2preds, nrow=38, ncol=length(lay2preds)/38)))
samp <- read_csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_v16preds.csv", row.names=FALSE)
