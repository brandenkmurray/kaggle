library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
# setwd("C:/Users/bmurray/My Documents/kaggle/telstra")
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <- data.table(read.csv("./data_trans/ts1Trans_v3.csv"))
xgbImpVars <- data.table(read_csv("./stack_models/xgb4Imp.csv"))
load("./data_trans/cvFoldsList.rda")


pairs <- combn(xgbImpVars$Feature[order(xgbImpVars$Frequence, decreasing = TRUE)][1:10], 2, simplify = FALSE)
for (i in 1:length(pairs)){
  name <- paste0(pairs[[i]][1],"_",pairs[[i]][2], "_impFreqInt")
  ts1Trans[,name] <- ts1Trans[,pairs[[i]][1], with=FALSE] / ts1Trans[,pairs[[i]][2], with=FALSE]
}

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter"), with=FALSE]))

# set.seed(201601)
# tsne_feats <- Rtsne(data.matrix(ts1Trans[,varnames,with=FALSE]), dims=2, initial_dims = 200, perplexity=80, theta=0.1, check_duplicates=TRUE, max_iter=500, verbose=TRUE)
# tsne_Y <- as.data.frame(tsne_feats$Y)
# colnames(tsne_Y) <- c("tsne_1", "tsne_2")
# write.csv(tsne_Y, "./stack_models/tsne_xgb6.csv", row.names=FALSE)
# tsne_Y$target <- as.factor(make.names(ts1Trans$target))
# (gg <- ggplot(tsne_Y[ts1Trans$filter==0,], aes(x=tsne_1, y=tsne_2, colour=target)) + geom_point(size=1))
# 
# ts1Trans <- cbind(ts1Trans, tsne_Y[,1:2])

dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))
#   eta md mcw   ss csbt
# 0.003 13  20 0.75  0.5  -- 0.454315+0.001111
# 0.003 13   5 0.75  0.5 -- 0.452639+0.000925

param <- list(objective="binary:logistic",
              eval_metric="logloss",
              eta = .003,
              max_depth=13,
              min_child_weight=5,
              subsample=.75,
              colsample_bytree=.5,
              nthread=7
)

set.seed(201512)
(tme <- Sys.time())
xgb6cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 40000,
                 folds=cvFoldsList,
                 maximize=FALSE,
                 prediction=TRUE,
                 print.every.n = 50,
                 early.stop.round=200)
Sys.time() - tme
save(xgb6cv, file="./stack_models/xgb6cv.rda")

write.csv(data.frame(ID=ts1Trans[filter==0,"ID",with=FALSE], PredictedProb=xgb6cv$pred), "./stack_models/cvPreds/cvPreds_xgb6.csv", row.names=FALSE)

minLossRound <- which.min(xgb6cv$dt$test.logloss.mean)
rounds <- floor(minLossRound * 1.15)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201512)
(tme <- Sys.time())
xgb6full <- xgb.train(data = dtrain,
                      params = param,
                      nrounds = rounds,
                      maximize=FALSE,
                      print.every.n = 20)
Sys.time() - tme
save(xgb6full, file="./stack_models/xgb6full.rda")

preds <- predict(xgb6full, data.matrix(ts1Trans[filter==2, c(varnames), with=FALSE]))
submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=preds)
write.csv(submission, "./stack_models/testPreds/testPreds_xgb6.csv", row.names=FALSE)


xgb6Imp <- xgb.importance(feature_names = colnames(ts1Trans[filter==2, c(varnames), with=FALSE]), model=xgb6full)
write.csv(xgb6Imp, "./stack_models/xgb6Imp.csv", row.names=FALSE)
