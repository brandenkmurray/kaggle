library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(Rtsne)
library(ggplot2)
setwd("/media/branden/SSHD1/kaggle/bnp")
ts1Trans <-fread("./data_trans/ts2Trans_v20.csv")
# xgbImpVars <- data.table(read_csv("./stack_models/xgb21Imp.csv"))
load("./data_trans/cvFoldsList.rda")


# pairs <- combn(xgbImpVars$Feature[order(xgbImpVars$Frequence, decreasing = TRUE)][1:10], 2, simplify = FALSE)
# for (i in 1:length(pairs)){
#   name <- paste0(pairs[[i]][1],"_",pairs[[i]][2], "_impFreqInt")
#   ts1Trans[,name] <- ts1Trans[,pairs[[i]][1], with=FALSE] / ts1Trans[,pairs[[i]][2], with=FALSE]
# }

varnames <- c(names(ts1Trans[filter==0, !colnames(ts1Trans) %in% c("ID","target","filter","dummy","pred0"), with=FALSE]))
# varnames <- xgbImpVars$Feature[order(xgbImpVars$Gain, decreasing=T)][1:2000]
# set.seed(201601)
# tsne_feats <- Rtsne(data.matrix(ts1Trans[,varnames,with=FALSE]), dims=2, initial_dims = 200, perplexity=80, theta=0.1, check_duplicates=TRUE, max_iter=500, verbose=TRUE)
# tsne_Y <- as.data.frame(tsne_feats$Y)
# colnames(tsne_Y) <- c("tsne_1", "tsne_2")
# write.csv(tsne_Y, "./stack_models/tsne_xgb27.csv", row.names=FALSE)
# tsne_Y$target <- as.factor(make.names(ts1Trans$target))
# (gg <- ggplot(tsne_Y[ts1Trans$filter==0,], aes(x=tsne_1, y=tsne_2, colour=target)) + geom_point(size=1))
# 
# ts1Trans <- cbind(ts1Trans, tsne_Y[,1:2])

# dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0, c(varnames),with=FALSE]),label=data.matrix(ts1Trans$target[ts1Trans$filter==0]))
#   eta md mcw   ss csbt
# 0.003 13  20 0.75  0.5  -- 0.454315+0.001111
# 0.003 13   5 0.75  0.5 -- 0.452639+0.000925
  
param <- list(objective="binary:logistic",
                eval_metric="logloss",
                eta = .01,
                max_depth=7,
                min_child_weight=1,
                subsample=.8,
                colsample_bytree=.4,
                nthread=13
  )


foldPreds <- rep(0,times = nrow(ts1Trans[ts1Trans$filter==0,]))
testPreds <- rep(0, times = nrow(ts1Trans[ts1Trans$filter==2,]) )
for (i in 1:length(cvFoldsList)) {
  print(paste0("Fold ",i))
  dtrain <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][-cvFoldsList[[i]], c(varnames),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][-cvFoldsList[[i]]]))
  dval <- xgb.DMatrix(data=data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(varnames),with=FALSE]),label=data.matrix(ts1Trans[filter==0, target][cvFoldsList[[i]]]))
  watch <- list(val=dval, train=dtrain)
    
  set.seed(201603)
  (tme <- Sys.time())
  xgb27 <- xgb.train(data = dtrain,
                     params = param,
                     nrounds = 50000,
                     maximize=FALSE,
                     watchlist=watch,
                     print.every.n = 50,
                     early.stop.round=200)
  Sys.time() - tme
  
  foldPreds[cvFoldsList[[i]]] <- predict(xgb27, data.matrix(ts1Trans[filter==0][cvFoldsList[[i]], c(varnames),with=FALSE]))
  testPreds <- testPreds + predict(xgb27, data.matrix(ts1Trans[filter==2, c(varnames),with=FALSE]))
  
  rm(xgb27); gc()
  
}

testPreds_adj <- testPreds/length(cvFoldsList)

write_csv(data.frame(ID=ts1Trans$ID[ts1Trans$filter==0], PredictedProb=foldPreds), "./stack_models/cvPreds/cvPreds_xgb27.csv")

submission <- data.frame(ID=ts1Trans$ID[ts1Trans$filter==2], PredictedProb=testPreds_adj)
write_csv(submission, "./stack_models/testPreds/testPreds_xgb27.csv")



