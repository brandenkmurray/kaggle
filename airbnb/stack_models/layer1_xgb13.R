library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(doParallel)
library(caret)
setwd("/home/branden/Documents/kaggle/airbnb")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)
ts1Trans <- data.table(read.csv("./data_trans/ts1_merge_v11.csv"))
# xgbImpVars <- read_csv("./stack_models/xgb10Imp.csv")
load("./data_trans/cvFoldsList.rda")
#ndcg metric from air's script
ndcg5 <- function(preds, dtrain) {
  
  labels <- getinfo(dtrain,"label")
  num.class = 12
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(list(metric = "ndcg5", value = ndcg))
}

# varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))
# dval <- xgb.DMatrix(data=data.matrix(train[cvFoldsList[[1]],varnames, with=FALSE]),label=data.matrix(train$class[cvFoldsList[[1]]]))
# dtrain <- xgb.DMatrix(data=data.matrix(train[-cvFoldsList[[1]],varnames, with=FALSE]),label=data.matrix(train$class[-cvFoldsList[[1]]]))
# watchlist <- list(train=dtrain, val=dval)

# varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))

varnames <- names(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE])
zVars <- nearZeroVar(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], freqCut=9999999, names=TRUE)
vars <- setdiff(varnames, zVars)


dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, vars, with=FALSE]),label=ts1Trans$class[ts1Trans$filter==0])

param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=12,
              eta = .003,
              max_depth=5,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              nthread=threads
)


(tme <- Sys.time())
set.seed(201512)
xgb13 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 10000,
               folds=cvFoldsList,
               prediction=TRUE,
               print.every.n = 20,
               early.stop.round=100)
Sys.time() - tme
save(xgb13, file="./stack_models/xgb13.rda")

#

cvPreds <- xgb13$pred
classMap <- read_csv("./data_trans/classMap.csv")
cnames <- paste("xgb13", classMap$country_destination, sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1Trans[filter==0,"id",with=FALSE], cvPreds), "./stack_models/cvPreds_xgb13.csv", row.names=FALSE) 


rounds <- floor(which.min(xgb13$dt$test.mlogloss.mean) * 1.15)

(tme <- Sys.time())
set.seed(201512)
xgb13full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = rounds,
                      maximize=FALSE,
                      print.every.n = 20)
Sys.time() - tme
save(xgb13full, file="./stack_models/xgb13full.rda")


testPreds <- predict(xgb13full, Matrix(data.matrix(ts1Trans[filter==2, vars, with=FALSE])))
testPreds <- as.data.frame(matrix(testPreds, nrow=12))
classMap <- read_csv("./data_trans/classMap.csv")
rownames(testPreds) <- classMap$country_destination
write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb13.csv", row.names=FALSE)
testPreds_top5 <- as.vector(apply(testPreds, 2, function(x) names(sort(x)[12:8])))


# create submission 
# ids <- NULL
# for (i in 1:NROW(ts1Trans[filter==2,])) {
#   idx <- as.character(ts1Trans$id[ts1Trans$filter==2][i])
#   ids <- append(ids, rep(idx,5))
# }
idx = ts1Trans$id[ts1Trans$filter==2]
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/testPreds_xgb13.csv", quote=FALSE, row.names = FALSE)

xgb13Imp <- xgb.importance(feature_names=colnames(ts1Trans[filter==0, vars, with=FALSE]), model=xgb13full)
write_csv(xgb13Imp, "./stack_models/xgb13Imp.csv")
