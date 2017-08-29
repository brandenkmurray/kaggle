library(readr)
library(data.table)
setwd("/home/branden/Documents/kaggle/airbnb")
t1 <- data.table(read.csv("./train_users_2.csv"))
load("./stack_models/xgbLay2_v9.rda")
lay2_nn2_cv <- read_csv("./stack_models/cvPreds_lay2_nn1.csv")

lay2_xgb9_test <- read_csv("./stack_models/lay2PredsProbs_xgb_v9.csv")
lay2_nn2_test <- read_csv("./stack_models/testPredsProbs_lay2_nn1.csv")

destClass <- data.frame(country_destination=sort(unique(t1$country_destination)), class=seq(0,11))
t1sub <- t1[order(t1$id), c("id","country_destination"), with=FALSE]
t1sub <- merge(t1sub, destClass, by="country_destination", all.x=TRUE, sort=FALSE)

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

ndcg5_v2 <- function(preds, acts) {
  
  labels <- acts
  num.class = 12
  pred <- t(preds)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  x <- ifelse(top==labels,1,0)
  dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
  ndcg <- mean(apply(x,1,dcg))
  return(ndcg)
}

ndcg5_v2(xgbLay2_v9$pred, as.numeric(t1sub$class))
ndcg5_v2(lay2_nn2_cv[,2:ncol(lay2_nn2_cv)], as.numeric(t1sub$class))

blendFrame <- (xgbLay2_v9$pred + lay2_nn2_cv[,2:ncol(lay2_nn2_cv)])/2

ndcg5_v2(blendFrame, t1sub$class)

blendFrameTest <- (lay2_xgb9_test[,2:ncol(lay2_xgb9_test)] + lay2_nn2_test[,2:ncol(lay2_nn2_test)])/2


classMap <- read_csv("./data_trans/classMap.csv")
sampID <- read_csv('sample_submission_NDF.csv')$id
sampID <- sort(sampID)
testPreds_top5 <- as.vector(apply(blendFrameTest, 1, function(x) names(sort(x)[12:8])))


# create submission 
idx = sampID
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/blended_v1.csv", quote=FALSE, row.names = FALSE)

