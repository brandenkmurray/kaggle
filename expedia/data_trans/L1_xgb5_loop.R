library(data.table)
library(Matrix)
library(gtools)
library(Metrics)
library(xgboost)
library(doParallel)
# setwd("/media/branden/SSHD1/kaggle/expedia")
# setwd("/media/branden/SSHD1/kaggle/expedia")
setwd("~/ebs")
# load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
map5 <- function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = t(matrix(preds, ncol = length(labels)))
  preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
  return (list(metric = 'map5', value = map5))
}
#######################
## Load data
#######################
ts1 <-  fread("./data_trans/ts1_merged_v1.csv")


varnames <- colnames(ts1)[!colnames(ts1) %in% c("cnt","id","id2","hotel_cluster","filter","dummy","pred0","date_time", "srch_ci","srch_co","date")]
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=100,
              eta = .1,
              max_depth=5,
              min_child_weight=1,
              subsample=.5,
              colsample_bytree=.3,
              nthread=threads)

dtest <- xgb.DMatrix(data=data.matrix(ts1[filter==2,varnames, with=FALSE]))



loops <- 20
testPredsMatrix <- matrix(0, nrow=sum(ts1$filter==2), ncol=100)
(tme <- Sys.time())
for (i in 1:loops){
    print(paste("loop ", i))
    set.seed(2050+i)
    samp <- sample(1:sum(ts1$filter %in% c(0,1)), 1000000, replace=FALSE)
    dtrain <- xgb.DMatrix(data=data.matrix(ts1[filter %in% c(0,1),varnames, with=FALSE][samp]),label=data.matrix(ts1[filter %in% c(0,1), hotel_cluster][samp]))
    # watchlist <- list(dtrain=dtrain)
    # rm(ts1)
    # gc()
    

    set.seed(201606+i)
    xgb1 <- xgb.train(data = dtrain,
                        params = param,
                        nrounds = 320
                        # folds=cvFoldsList,
                        # maximize=FALSE,
                        # prediction=TRUE,
                        # watchlist=watchlist,
                        # early.stop.round=7,
                        # print.every.n = 10
                        )


    testPreds <- predict(xgb1, dtest)
    testPreds <- as.data.table(t(matrix(testPreds, nrow=100)))
    colnames(testPreds) <- as.character(0:99)
    fwrite(data.table(id=seq(0,nrow(testPreds)-1,1),testPreds), paste0("./stack_models/L1/testPreds/L1_testPreds_xgb5_loop",i,"_probs.csv"))
    
    testPredsMatrix <- testPredsMatrix + testPreds
}  
Sys.time() - tme   

testPredsFinal <- testPredsMatrix/loops
colnames(testPredsFinal) <- as.character(0:99)
testPreds_top5 <- as.data.frame(t(apply(testPredsFinal, 1, function(x) names(sort(x, decreasing=T)[1:5]))))
testPreds_top5_concat <- do.call("paste", c(testPreds_top5, sep=" "))
    
submission <- data.table(id=seq(0,nrow(testPreds_top5)-1,1), hotel_cluster=testPreds_top5_concat)
    
fwrite(submission, "./stack_models/L1/testPreds/L1_testPreds_xgb5_loop.csv")
fwrite(data.table(id=seq(0,nrow(testPreds_top5)-1,1),testPreds), "./stack_models/L1/testPreds/L1_testPreds_xgb5_loop_probs.csv")


# create_feature_map <- function(fmap_filename, features){
#   for (i in 1:length(features)){
#     cat(paste(c(i-1,features[i],"q"), collapse = "\t"), file=fmap_filename, sep="\n",append=TRUE)
#   }
# }
# create_feature_map("./xgb1_fmap.txt", varnames)
# 
# xgb.dump(model=xgb1, fname="xgb_dump",fmap="./xgb1_fmap.txt", with.stats = TRUE)
# xgb1Imp <- xgb.importance(feature_names=varnames, model=xgb1)
# View(xgb1Imp)
# 



# L1_testPreds_xgb1 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1.csv")
# 
# 
# 
# 
# leak <- leak[ts1$filter==2]
# leak_rowSums <- rowSums(leak[,2:ncol(leak),with=F])
# leak_rows <- leak_rowSums>=0
# 
# colnames(leak) <- c("id2",as.character(0:99))
# # write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb11.csv", row.names=FALSE)
# testPreds_top5_leak <- as.data.frame(t(apply(leak[,2:ncol(leak),with=F], 1, function(x) names(sort(x, decreasing=T)[1:5]))))
# 
# testPreds_top5_concat <- do.call("paste", c(testPreds_top5_leak, sep=" "))
# testPreds_top5_leak_frame <- data.table(id2=leak$id2, hotel_cluster=testPreds_top5_concat)
# 
# 
# L1_testPreds_xgb1$hotel_cluster[leak_rows] <- testPreds_top5_leak_frame$hotel_cluster[leak_rows]
# fwrite(L1_testPreds_xgb1, "./stack_models/L1/testPreds/L1_testPreds_xgb1_leaktest.csv")
