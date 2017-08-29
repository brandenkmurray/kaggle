library(xgboost)
library(caret)
setwd("/media/branden/SSHD1/kaggle/bnp")

t1 <- read.csv("./train.csv")
s1 <- read.csv("./test.csv")

t1[is.na(t1)] <- -1
s1[is.na(s1)] <- -1

# factors <- colnames(t1)[sapply(t1, is.factor)]
# fewFacts <- colnames(t1[factors])[sapply(t1[factors], function(x) length(unique(x))<40)]

# uniq <- sapply(t1[,3:ncol(t1)], function(x) length(unique(x)))
# fewUniq <- names(uniq)[uniq < 40]


# varnames <- c(fewFacts, fewUniq)

set.seed(2016)
cvFolds <- createFolds(t1$target, k=5, list=TRUE, returnTrain=FALSE)
dtrain<-xgb.DMatrix(data=data.matrix(sapply(t1[,!colnames(t1) %in% c("ID","target")], as.numeric)),label=data.matrix(t1$target))

# Train Model
param <- list(booster="gbtree",
              eta=0.05,
              max_depth=5,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=.6,
              objective="binary:logistic",
              eval_metric="logloss")


set.seed(201510)
(tme <- Sys.time())
xgb1cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 5000,
                 maximize=FALSE,
                 # watchlist=watchlist,
                 folds=cvFolds,
                 print.every.n = 10,
                 early.stop.round=100)
Sys.time() - tme
save(xgb1cv, file="./stack_models/xgb1cv.rda")

rounds <- round(which.min(xgb1cv$test.logloss.mean)*1.15)

(tme <- Sys.time())
xgb1 <- xgb.train(data = dtrain,
                  params = param,
                  nrounds = rounds,
                  # maximize=FALSE,
                  # watchlist=watchlist,
                  # folds=cvFolds,
                  print.every.n = 1)
Sys.time() - tme


testPreds <- predict(xgb1, s1[,!colnames(s1) %in% c("ID","target")])
submission <- data.frame(ID=s1$ID, PredictProb=testPreds)
write_csv(submission, "./stack_models/testPreds/testPreds_xgb1.csv")

(imp <- xgb.importance(feature_names = names(t1[,fewUniq]), model=xgb1))
