library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
setwd("/home/branden/Documents/kaggle/walmart")

# Load CV predictions from models
xgb1preds <- read_csv("./stack_models/cvPreds_xgb1.csv")
xgb2preds <- read_csv("./stack_models/cvPreds_xgb2.csv")
xgb3preds <- read_csv("./stack_models/cvPreds_xgb3.csv")
xgb7preds <- read_csv("./stack_models/cvPreds_xgb7.csv")
xgb8preds <- read_csv("./stack_models/cvPreds_xgb8.csv")
xgb9preds <- read_csv("./stack_models/cvPreds_xgb9.csv")
xgb10preds <- read_csv("./stack_models/cvPreds_xgb10.csv")
xgb11preds <- read_csv("./stack_models/cvPreds_xgb11.csv")
kknn1preds <- read_csv("./stack_models/cvPreds_kknn1.csv")
kknn2preds <- read_csv("./stack_models/cvPreds_kknn2.csv")
kknn3preds <- read_csv("./stack_models/cvPreds_kknn3.csv")
kknn4preds <- read_csv("./stack_models/cvPreds_kknn4.csv")
nn1preds <- read_csv("./stack_models/cvPreds_nn1.csv")
nn2preds <- read_csv("./stack_models/cvPreds_nn2.csv")
et1preds <- read_csv("./stack_models/cvPreds_et1.csv")
glmnet1preds <- read_csv("./stack_models/cvPreds_glmnet1.csv")
glmnet2preds <- read_csv("./stack_models/cvPreds_glmnet2.csv")
# Edit and bind predictions
xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
xgb3preds$VisitNumber <- NULL
xgb7preds$VisitNumber <- NULL
xgb8preds$VisitNumber <- NULL
xgb9preds$VisitNumber <- NULL
xgb10preds$VisitNumber <- NULL
xgb11preds$VisitNumber <- NULL
kknn1preds$VisitNumber <- NULL
kknn2preds$VisitNumber <- NULL
kknn3preds$VisitNumber <- NULL
kknn4preds$VisitNumber <- NULL
nn1preds$VisitNumber <- NULL
nn2preds$VisitNumber <- NULL
et1preds$VisitNumber <- NULL
glmnet1preds$VisitNumber <- NULL
glmnet2preds$VisitNumber <- NULL
glmnet1preds[is.na(glmnet1preds)] <- 0
glmnet2preds[is.na(glmnet2preds)] <- 0
lay1preds <- cbind(xgb1preds, xgb7preds, xgb8preds, xgb9preds, xgb11preds, kknn1preds, kknn4preds, nn1preds, nn2preds, et1preds, glmnet1preds, glmnet2preds)
# lay1preds <- cbind(xgb1preds,xgb7preds, kknn1preds, nn1preds, nn2preds)
# Add the class column to the dataset
t1 <- data.table(read_csv("train.csv"))
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

load("./data_trans/cvFoldsList_lay2.rda")
dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[,"class", with=FALSE]))



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
              eta = .02,
              max_depth=3,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              nthread=12
)
set.seed(201510)
(tme <- Sys.time())
xgbLay2_v16 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               prediction=TRUE,
               folds=cvFoldsList_lay2,
               # watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgbLay2_v16, file="./stack_models/xgbLay2_v16.rda")

rounds <- floor(which.min(xgbLay2_v16$dt$test.mlogloss.mean) * 1.175)


# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read_csv("./stack_models/testPreds_xgb1full.csv")
xgb2fullpreds <- read_csv("./stack_models/testPreds_xgb2full.csv")
xgb3fullpreds <- read_csv("./stack_models/testPreds_xgb3full.csv")
xgb7fullpreds <- read_csv("./stack_models/testPreds_xgb7full.csv")
xgb8fullpreds <- read_csv("./stack_models/testPreds_xgb8full.csv")
xgb9fullpreds <- read_csv("./stack_models/testPreds_xgb9full.csv")
xgb10fullpreds <- read_csv("./stack_models/testPreds_xgb10full.csv")
xgb11fullpreds <- read_csv("./stack_models/testPreds_xgb10full.csv")
kknn1fullpreds <- read_csv("./stack_models/testPreds_kknn1full.csv")
kknn2fullpreds <- read_csv("./stack_models/testPreds_kknn2full.csv")
kknn3fullpreds <- read_csv("./stack_models/testPreds_kknn3full.csv")
kknn4fullpreds <- read_csv("./stack_models/testPreds_kknn4full.csv")
nn1fullpreds <- read_csv("./stack_models/testPreds_nn1full.csv")
nn2fullpreds <- read_csv("./stack_models/testPreds_nn2full.csv")
et1fullpreds <- read_csv("./stack_models/testPreds_et1full.csv")
glmnet1fullpreds <- read_csv("./stack_models/testPreds_glmnet1full.csv")
glmnet2fullpreds <- read_csv("./stack_models/testPreds_glmnet2full.csv")
glmnet1fullpreds[is.na(glmnet1fullpreds)] <- 0
glmnet2fullpreds[is.na(glmnet2fullpreds)] <- 0
# Edit and bind test set predictions
xgb1fullpreds$VisitNumber <- NULL
xgb2fullpreds$VisitNumber <- NULL
xgb3fullpreds$VisitNumber <- NULL
xgb7fullpreds$VisitNumber <- NULL
xgb8fullpreds$VisitNumber <- NULL
xgb9fullpreds$VisitNumber <- NULL
xgb10fullpreds$VisitNumber <- NULL
xgb11fullpreds$VisitNumber <- NULL
kknn1fullpreds$VisitNumber <- NULL
kknn2fullpreds$VisitNumber <- NULL
kknn3fullpreds$VisitNumber <- NULL
kknn4fullpreds$VisitNumber <- NULL
nn1fullpreds$VisitNumber <- NULL
nn2fullpreds$VisitNumber <- NULL
et1fullpreds$VisitNumber <- NULL
glmnet1fullpreds$VisitNumber<- NULL
glmnet2fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb7fullpreds, xgb8fullpreds, xgb9fullpreds, xgb11fullpreds, kknn1fullpreds, kknn4fullpreds,nn1fullpreds, nn2fullpreds, et1fullpreds, glmnet1fullpreds,glmnet2fullpreds)
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
