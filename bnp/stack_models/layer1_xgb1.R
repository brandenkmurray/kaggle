library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
# setwd("C:/Users/bmurray/My Documents/kaggle/telstra")
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <- data.table(read.csv("./data_trans/ts1Trans_v1.csv"))
# xgbImpVars <- data.table(read_csv("./stack_models/xgb7Imp.csv"))
load("./data_trans/cvFoldsList.rda")
varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter"), with=FALSE]))

# set.seed(201601)
# tsne_feats <- Rtsne(data.matrix(ts1Trans[,varnames,with=FALSE]), dims=2, initial_dims = 100, perplexity=50, theta=0.1, check_duplicates=TRUE, max_iter=1500, verbose=TRUE)
# tsne_Y <- as.data.frame(tsne_feats$Y)
# colnames(tsne_Y) <- c("tsne_1", "tsne_2")
# write.csv(tsne_Y, "./stack_models/tsne_xgb1.csv")
# tsne_Y$target <- as.factor(make.names(ts1Trans$target))
# (gg <- ggplot(tsne_Y[ts1Trans$filter==0,], aes(x=tsne_1, y=tsne_2, colour=target)) + geom_point())

# ts1Trans <- cbind(ts1Trans, tsne_Y[,1:2])

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))

param <- list(objective="binary:logistic",
              eval_metric="logloss",
              eta = .05,
              max_depth=7,
              min_child_weight=1,
              subsample=.7,
              colsample_bytree=.7,
              nthread=7
)

set.seed(201512)
(tme <- Sys.time())
xgb1cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 30000,
                 folds=cvFoldsList,
                 maximize=FALSE,
                 prediction=TRUE,
                 print.every.n = 50,
                 early.stop.round=200)
Sys.time() - tme
save(xgb1cv, file="./stack_models/xgb1cv.rda")


samp <- read.csv('sample_submission.csv')
cnames <- paste("xgb1", names(samp)[2], sep="_")
colnames(xgb1cv$pred) <- cnames
write.csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], PredictedProb=xgb1cv$pred), "./stack_models/cvPreds/cvPreds_xgb1.csv", row.names=FALSE)

minLossRound <- which.min(xgb1cv$dt$test.logloss.mean)
rounds <- floor(minLossRound * 1.15)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201512)
(tme <- Sys.time())
xgb1full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = rounds,
                      maximize=FALSE,
                      print.every.n = 20)
Sys.time() - tme
save(xgb1full, file="./stack_models/xgb1full.rda")

preds <- predict(xgb1full, data.matrix(ts1Trans[filter==2, c(varnames), with=FALSE]))
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb1.csv", row.names=FALSE)


xgb1Imp <- xgb.importance(feature_names = colnames(ts1Trans[filter==2, c(varnames), with=FALSE]), model=xgb1full)
write.csv(xgb1Imp, "./stack_models/xgb1Imp.csv", row.names=FALSE)
