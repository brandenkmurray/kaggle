library(xgboost)
library(data.table)
library(readr)
library(Matrix)
setwd("/home/branden/Documents/kaggle/airbnb")
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v3.csv"))
xgbImpVars <- read_csv("./stack_models/xgb5Imp.csv")
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

varnames<- c("daysDiffCreatedAct",xgbImpVars$Feature)
dtrain <- xgb.DMatrix(data=Matrix(data.matrix(ts1Trans[filter==0,varnames, with=FALSE])),label=ts1Trans$class[ts1Trans$filter==0])

param <- list(objective="multi:softprob",
              eval_metric=ndcg5,
              num_class=12,
              eta = .005,
              max_depth=10,
              min_child_weight=1,
              subsample=.5,
              colsample_bytree=.5
)


(tme <- Sys.time())
set.seed(201512)
xgb17 <- xgb.cv(data = dtrain,
               params = param,
               nrounds = 5000,
               folds=cvFoldsList,
               prediction=TRUE,
               maximize=TRUE,
               print.every.n = 20,
               early.stop.round=100)
Sys.time() - tme
save(xgb17, file="./stack_models/xgb17.rda")

cvPreds <- xgb17$pred
classMap <- read_csv("./data_trans/classMap.csv")
cnames <- paste("xgb17", classMap$country_destination, sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1Trans[filter==0,"id",with=FALSE], cvPreds), "./stack_models/cvPreds_xgb17.csv", row.names=FALSE) 


rounds <- floor(which.max(xgb17$dt$test.ndcg5.mean) * 1.2)

(tme <- Sys.time())
set.seed(201512)
xgb17full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = rounds,
                      maximize=FALSE,
                      print.every.n = 20)
Sys.time() - tme
save(xgb17full, file="./stack_models/xgb17full.rda")


testPreds <- predict(xgb17full, data.matrix(ts1Trans[filter==2,c("daysDiffCreatedAct",xgbImpVars$Feature), with=FALSE]))
testPreds <- as.data.frame(matrix(testPreds, nrow=12))
classMap <- read_csv("./data_trans/classMap.csv")
rownames(testPreds) <- classMap$country_destination
write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb17.csv", row.names=FALSE)
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
write.csv(submission, "./stack_models/testPreds_xgb17.csv", quote=FALSE, row.names = FALSE)
