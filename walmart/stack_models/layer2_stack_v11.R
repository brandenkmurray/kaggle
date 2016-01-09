library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
setwd("/home/branden/Documents/kaggle/walmart")

# Load CV predictions from models
xgb1preds <- read.csv("./stack_models/cvPreds_xgb1.csv")
xgb2preds <- read.csv("./stack_models/cvPreds_xgb2.csv")
xgb3preds <- read.csv("./stack_models/cvPreds_xgb3.csv")
xgb7preds <- read.csv("./stack_models/cvPreds_xgb7.csv")
kknn1preds <- read.csv("./stack_models/cvPreds_kknn1.csv")
kknn2preds <- read.csv("./stack_models/cvPreds_kknn2.csv")
nn1preds <- read.csv("./stack_models/cvPreds_nn1.csv")
nn2preds <- read.csv("./stack_models/cvPreds_nn2.csv")
# Edit and bind predictions
xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
xgb3preds$VisitNumber <- NULL
xgb7preds$VisitNumber <- NULL
kknn1preds$VisitNumber <- NULL
kknn2preds$VisitNumber <- NULL
nn1preds$VisitNumber <- NULL
nn2preds$VisitNumber <- NULL
lay1preds <- cbind(xgb1preds, xgb2preds, xgb3preds, xgb7preds, kknn1preds, nn1preds, nn2preds)
# lay1preds <- cbind(xgb1preds,xgb7preds, kknn1preds, nn1preds, nn2preds)
# Add the class column to the dataset
t1 <- data.table(read.csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class)]

lay1preds <- data.table(cbind(class=t1$class, lay1preds))

# Create a validation set
# set.seed(1234)
# h <- sample(nrow(lay1preds), 2000)
# # Create DMatrices
# dval <- xgb.DMatrix(data=data.matrix(lay1preds[h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[h,"class", with=FALSE]))
# dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[-h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[-h,"class", with=FALSE]))
# watchlist <- list(val=dval,train=dtrain)

set.seed(1234)
cvFolds <- createFolds(lay1preds$class, k=4, list=TRUE, returnTrain=FALSE)
dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[,"class", with=FALSE]))


# Train Model
# xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.564680+0.003101
# xgb1, xgb2, xgb3, kknn1, nn1 = 0.567281+0.002696
# xgb1, xgb7, kknn1, nn1, nn2 = 0.562542+0.003631
# subsample=0.7, colsample=0.7, xgb1, xgb2, xgb3, xgb7, kknn1, nn1, nn2 = 0.559679+0.002955
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .05,
              max_depth=3,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7
)
set.seed(201510)
(tme <- Sys.time())
xgbLay2_v11 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               folds=cvFolds,
               # watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgbLay2_v11, file="./stack_models/xgbLay2_v11.rda")



# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read.csv("./stack_models/testPreds_xgb1full.csv")
xgb2fullpreds <- read.csv("./stack_models/testPreds_xgb2full.csv")
xgb3fullpreds <- read.csv("./stack_models/testPreds_xgb3full.csv")
xgb7fullpreds <- read.csv("./stack_models/testPreds_xgb7full.csv")
kknn1fullpreds <- read.csv("./stack_models/testPreds_kknn1full.csv")
kknn2fullpreds <- read.csv("./stack_models/testPreds_kknn2full.csv")
nn1fullpreds <- read.csv("./stack_models/testPreds_nn1full.csv")
nn2fullpreds <- read.csv("./stack_models/testPreds_nn2full.csv")
# Edit and bind test set predictions
xgb1fullpreds$VisitNumber <- NULL
xgb2fullpreds$VisitNumber <- NULL
xgb3fullpreds$VisitNumber <- NULL
xgb7fullpreds$VisitNumber <- NULL
kknn1fullpreds$VisitNumber <- NULL
kknn2fullpreds$VisitNumber <- NULL
nn1fullpreds$VisitNumber <- NULL
nn2fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds, xgb3fullpreds, xgb7fullpreds, kknn1fullpreds, nn1fullpreds, nn2fullpreds)
# lay1fullpreds <- cbind(xgb1fullpreds,xgb7fullpreds, kknn1fullpreds, nn1fullpreds, nn2fullpreds)
# Predict the test set using the XGBOOST stacked model

set.seed(201510)
(tme <- Sys.time())
xgbLay2_v11_full <- xgb.train(data = dtrain,
                        params = param,
                        nrounds = 275,
                        maximize=FALSE,
                        # watchlist=watchlist,
                        print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v11_full, file="./stack_models/xgbLay2_v11_full.rda")



lay2preds <- predict(xgbLay2_v11_full, newdata=data.matrix(lay1fullpreds))
preds <- data.frame(t(matrix(lay2preds, nrow=38, ncol=length(lay2preds)/38)))
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_v11_preds.csv", row.names=FALSE)
