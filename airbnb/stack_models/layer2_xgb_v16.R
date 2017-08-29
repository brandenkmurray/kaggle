library(data.table)
library(readr)
library(xgboost)
library(doParallel)
setwd("/home/branden/Documents/kaggle/airbnb")
load("./data_trans/cvFoldsList_lay2_k6.rda")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)

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

xgb1cv <- read_csv("./stack_models/cvPreds_xgb1.csv") 
xgb2cv <- read_csv("./stack_models/cvPreds_xgb2.csv") 
xgb3cv <- read_csv("./stack_models/cvPreds_xgb3.csv") 
xgb4cv <- read_csv("./stack_models/cvPreds_xgb4.csv") 
xgb5cv <- read_csv("./stack_models/cvPreds_xgb5.csv") 
xgb6cv <- read_csv("./stack_models/cvPreds_xgb6.csv")
xgb7cv <- read_csv("./stack_models/cvPreds_xgb7.csv")
xgb8cv <- read_csv("./stack_models/cvPreds_xgb8.csv")
xgb9cv <- read_csv("./stack_models/cvPreds_xgb9.csv")
xgb10cv <- read_csv("./stack_models/cvPreds_xgb10.csv")
xgb11cv <- read_csv("./stack_models/cvPreds_xgb11.csv")
xgb12cv <- read_csv("./stack_models/cvPreds_xgb12.csv")
xgb13cv <- read_csv("./stack_models/cvPreds_xgb13.csv")
xgb14cv <- read_csv("./stack_models/cvPreds_xgb14.csv")
xgb15cv <- read_csv("./stack_models/cvPreds_xgb15.csv")
xgb17cv <- read_csv("./stack_models/cvPreds_xgb17.csv")
nn1cv <- read_csv("./stack_models/cvPreds_nn1.csv") 
colnames(nn1cv)[2:13] <- paste0("nn1_",colnames(nn1cv)[2:13] )
nn2cv <- read_csv("./stack_models/cvPreds_nn2.csv") 
colnames(nn2cv)[2:13] <- paste0("nn2_",colnames(nn2cv)[2:13] )
nn3cv <- read_csv("./stack_models/cvPreds_nn3.csv") 
colnames(nn3cv)[2:13] <- paste0("nn3_",colnames(nn3cv)[2:13] )
nn4cv <- read_csv("./stack_models/cvPreds_nn4.csv") 
colnames(nn4cv)[2:13] <- paste0("nn4_",colnames(nn4cv)[2:13] )
glmnet1cv <- read_csv("./stack_models/cvPreds_glmnet1.csv") 
rf1cv <- read_csv("./stack_models/cvPreds_rf1.csv")

xgb1cv$id <- NULL
xgb2cv$id <- NULL
xgb3cv$id <- NULL
xgb4cv$id <- NULL
xgb5cv$id <- NULL
xgb6cv$id <- NULL
xgb7cv$id <- NULL
xgb8cv$id <- NULL
xgb9cv$id <- NULL
xgb10cv$id <- NULL
xgb11cv$id <- NULL
xgb12cv$id <- NULL
xgb13cv$id <- NULL
xgb14cv$id <- NULL
xgb15cv$id <- NULL
xgb17cv$id <- NULL
nn1cv$id <- NULL
nn2cv$id <- NULL
nn3cv$id <- NULL
nn4cv$id <- NULL
glmnet1cv$id <- NULL
rf1cv$id <- NULL
lay1preds <- cbind(xgb1cv, xgb2cv, xgb4cv, xgb6cv, xgb7cv, xgb8cv, xgb9cv, xgb10cv, xgb11cv, xgb12cv, xgb13cv, xgb14cv, xgb15cv, xgb17cv, nn1cv, nn2cv, nn3cv, nn4cv, glmnet1cv, rf1cv)
# Get actual classes
t1 <- data.table(read.csv("./train_users_2.csv"))
classMap <- read_csv("./data_trans/classMap.csv")
t1 <- merge(t1, classMap, by="country_destination")
t1 <- t1[order(t1$id),]
country_destination<- t1$country_destination

lay1preds <- data.table(cbind(class=t1$class, lay1preds))

dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[,"class", with=FALSE]))



# Train Model
# eta=0.01, md=3, mcw=1, ss=0.7, csbt=0.7, xgb1, xgb2, xgb3, xgb4, xgb5, xgb6 = 1.022082+0.004287
# eta=0.005, md=4, mcw=1,ss=0.5, csbt=0.5, xgb1, xgb2, xgb3, xgb4, xgb5, xgb6 =1.021717 + 0.00433
# eta=0.005, md=4, mcw=1,ss=0.7, csbt=0.7, xgb1, xgb2, xgb3, xgb4, xgb5, xgb6, nn1, glmnet1 =1.020788+0.004354
# eta=0.005, md=4, mcw=1,ss=0.7, csbt=0.7, xgb6, xgb7, xgb8, xgb9, xgb10, xgb11, nn1, nn2, nn3, glmnet1, rf1 =1.020613+0.004392
# eta=0.005, md=4, mcw=1,ss=0.7, csbt=0.7, xgb6, xgb7, xgb8, xgb9, xgb10, xgb11, xgb12, xgb13, nn1, nn2, nn3, glmnet1, rf1 =1.02046+0.004225
# lay2_v9 eta=0.005, md=5, mcw=1,ss=0.5, csbt=0.5, xgb1, xgb4 (old),xgb6, xgb7, xgb8, xgb9, xgb10, xgb11, xgb12, xgb13, xgb14, nn1, nn2, nn3, glmnet1, rf1 =0.832915 +0.000795
# lay2_v13 eta=0.005, md=5, mcw=1,ss=0.5, csbt=0.5, xgb1,xgb2,xgb4 (old),xgb6, xgb7, xgb8, xgb9, xgb10, xgb11, xgb12, xgb13, xgb14, nn1, nn2, nn3, nn4, glmnet1, rf1 =0.833184+0.001747
# eta=0.005, md=5, mcw=1,ss=0.5, csbt=0.5, xgb1,xgb2,xgb4,xgb6, xgb7, xgb8, xgb9, xgb10, xgb11, xgb12, xgb13, xgb14,xgb15, nn1, nn2, nn3, nn4, glmnet1, rf1 =1.02046+0.004225

param <- list(objective="multi:softprob",
              eval_metric=ndcg5,
              num_class=12,
              eta = .005,
              max_depth=5,
              min_child_weight=1,
              subsample=.5,
              colsample_bytree=.5,
              nthread=threads
)

# xgb14  = 0.83287+0.000968

set.seed(201510)
(tme <- Sys.time())
xgbLay2_v16 <- xgb.cv(data = dtrain,
                      params = param,
                      nrounds = 10000,
                      # feval = ndcg5,
                      maximize=TRUE,
                      prediction=TRUE,
                      folds=cvFoldsList_lay2_k6,
                      # watchlist=watchlist,
                      print.every.n = 1,
                      early.stop.round=100)
Sys.time() - tme
save(xgbLay2_v16, file="./stack_models/xgbLay2_v16.rda")

xgbLay2_v16$dt[which.max(xgbLay2_v16$dt$test.ndcg5.mean),]
rounds <- floor(which.max(xgbLay2_v16$dt$test.ndcg5.mean) * 1.15)

# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read_csv("./stack_models/testPredsProbs_xgb1.csv")
xgb2fullpreds <- read_csv("./stack_models/testPredsProbs_xgb2.csv")
xgb3fullpreds <- read_csv("./stack_models/testPredsProbs_xgb3.csv")
xgb4fullpreds <- read_csv("./stack_models/testPredsProbs_xgb4.csv")
xgb5fullpreds <- read_csv("./stack_models/testPredsProbs_xgb5.csv")
xgb6fullpreds <- read_csv("./stack_models/testPredsProbs_xgb6.csv")
xgb7fullpreds <- read_csv("./stack_models/testPredsProbs_xgb7.csv")
xgb8fullpreds <- read_csv("./stack_models/testPredsProbs_xgb8.csv")
xgb9fullpreds <- read_csv("./stack_models/testPredsProbs_xgb9.csv")
xgb10fullpreds <- read_csv("./stack_models/testPredsProbs_xgb10.csv")
xgb11fullpreds <- read_csv("./stack_models/testPredsProbs_xgb11.csv")
xgb12fullpreds <- read_csv("./stack_models/testPredsProbs_xgb12.csv")
xgb13fullpreds <- read_csv("./stack_models/testPredsProbs_xgb13.csv")
xgb14fullpreds <- read_csv("./stack_models/testPredsProbs_xgb14.csv")
xgb15fullpreds <- read_csv("./stack_models/testPredsProbs_xgb15.csv")
xgb17fullpreds <- read_csv("./stack_models/testPredsProbs_xgb17.csv")
nn1fullpreds <- read_csv("./stack_models/testPredsProbs_nn1.csv")
nn2fullpreds <- read_csv("./stack_models/testPredsProbs_nn2.csv")
nn3fullpreds <- read_csv("./stack_models/testPredsProbs_nn3.csv")
nn4fullpreds <- read_csv("./stack_models/testPredsProbs_nn4.csv")
glmnet1fullpreds <- read_csv("./stack_models/testPredsProbs_glmnet1.csv")
rf1fullpreds <- read_csv("./stack_models/testPredsProbs_rf1.csv")
# Edit and bind test set predictions
xgb1fullpreds$id <- NULL
xgb2fullpreds$id <- NULL
xgb3fullpreds$id <- NULL
xgb4fullpreds$id <- NULL
xgb5fullpreds$id <- NULL
xgb6fullpreds$id <- NULL
xgb7fullpreds$id <- NULL
xgb8fullpreds$id <- NULL
xgb9fullpreds$id <- NULL
xgb10fullpreds$id <- NULL
xgb11fullpreds$id <- NULL
xgb12fullpreds$id <- NULL
xgb13fullpreds$id <- NULL
xgb14fullpreds$id <- NULL
xgb15fullpreds$id <- NULL
xgb17fullpreds$id <- NULL
nn1fullpreds$id <- NULL
nn2fullpreds$id <- NULL
nn3fullpreds$id <- NULL
nn4fullpreds$id <- NULL
glmnet1fullpreds$id <- NULL
rf1fullpreds$id <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds, xgb4fullpreds,xgb6fullpreds, xgb7fullpreds, xgb8fullpreds, xgb9fullpreds, xgb10fullpreds, xgb11fullpreds, xgb12fullpreds, xgb13fullpreds, xgb14fullpreds, xgb15fullpreds, xgb17fullpreds, nn1fullpreds, nn2fullpreds,nn3fullpreds, nn4fullpreds, glmnet1fullpreds,rf1fullpreds)
# Predict the test set using the XGBOOST stacked model

set.seed(201510)
(tme <- Sys.time())
xgbLay2_v16_full <- xgb.train(data = dtrain,
                              params = param,
                              nrounds = rounds,
                              maximize=TRUE,
                              # watchlist=watchlist,
                              print.every.n = 5)
Sys.time() - tme
save(xgbLay2_v16_full, file="./stack_models/xgbLay2_v16_full.rda")

lay2Preds <- predict(xgbLay2_v16_full, newdata=data.matrix(lay1fullpreds))
lay2Preds <- as.data.frame(matrix(lay2Preds, nrow=12))
classMap <- read_csv("./data_trans/classMap.csv")
rownames(lay2Preds) <- classMap$country_destination
sampID <- read_csv('sample_submission_NDF.csv')$id
sampID <- sort(sampID)
write.csv(data.frame(id=sampID, t(lay2Preds)), "./stack_models/lay2PredsProbs_xgb_v16.csv", row.names=FALSE)
testPreds_top5 <- as.vector(apply(lay2Preds, 2, function(x) names(sort(x)[12:8])))


# create submission 
idx = sampID
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/lay2Preds_xgb_v16.csv", quote=FALSE, row.names = FALSE)

xgbLay2v16Imp <- xgb.importance(feature_names = colnames(lay1preds[,2:ncol(lay1preds), with=FALSE]), model=xgbLay2_v16_full )
