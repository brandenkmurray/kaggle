library(readr)
library(data.table)
library(caret)
library(reshape2)
library(dplyr)
library(kknn)
library(Matrix)
library(doParallel)
library(gtools)
# library(h2o)
library(ranger)
setwd("/home/branden/Documents/kaggle/airbnb")
threads <- ifelse(detectCores()>8,detectCores()-3,detectCores()-1)
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v4.csv"))
# xgbImpVars <- read_csv("./stack_models/xgb7Imp.csv")
load("./data_trans/cvFoldsTrainList.rda")

# #ndcg metric from air's script
# ndcg5 <- function(preds, dtrain) {
#   
#   labels <- getinfo(dtrain,"label")
#   num.class = 12
#   pred <- matrix(preds, nrow = num.class)
#   top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
#   
#   x <- ifelse(top==labels,1,0)
#   dcg <- function(y) sum((2^y - 1)/log(2:(length(y)+1), base = 2))
#   ndcg <- mean(apply(x,1,dcg))
#   return(list(metric = "ndcg5", value = ndcg))
# }
# 
# # Logloss function
# LogLoss <- function(actual, predicted, eps=1e-15) {
#   predicted[predicted < eps] <- eps;
#   predicted[predicted > 1 - eps] <- 1 - eps;
#   -1/nrow(actual)*(sum(actual*log(predicted)))
# }

pp <- preProcess(ts1Trans[filter==0, 4:ncol(ts1Trans),with=FALSE], method=c("zv","center","scale"))
train <- predict(pp, ts1Trans[filter==0, 4:ncol(ts1Trans),with=FALSE])
test <- predict(pp, ts1Trans[filter==2, 4:ncol(ts1Trans),with=FALSE])
train_class <- as.factor(make.names(ts1Trans$class[ts1Trans$filter==0]))

rfControl <- trainControl(method="cv",
                          number=5,
                          summaryFunction=mnLogLoss,
                          savePredictions=TRUE,
                          classProbs=TRUE,
                          index=cvFoldsTrainList,
                          allowParallel=TRUE)
rfGrid <- expand.grid(mtry=c(13))

cl <- makeCluster(5)
registerDoParallel(cl)
set.seed(201601)
(tme <- Sys.time())
rf2 <- train(x=train,
             y=train_class,
             method="ranger",
             trControl=rfControl,
             num.trees=2000,
             min.node.size=1,
             # tuneLength=2,
             tuneGrid=rfGrid,
             metric="logLoss",
             save.memory=TRUE)
stopCluster(cl)
Sys.time() - tme
save(rf2, file="./stack_models/layer1_rf2.rda")

cvPreds <- rf2$pred[,3:15]
cvPreds <- cvPreds[order(cvPreds$rowIndex),mixedorder(names(cvPreds))]
cvPreds$rowIndex <- NULL

samp <- read_csv('sample_submission_NDF.csv')
cnames <- paste("rf2", names(cvPreds)[1:ncol(cvPreds)], sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1Trans[filter==0,"id",with=FALSE], cvPreds), "./stack_models/cvPreds_rf2.csv", row.names=FALSE) 


preds <- predict(rf2, newdata=test, type="prob")
preds <- preds[,mixedorder(names(preds))]
samp <- read_csv('sample_submission_NDF.csv')
classMap <- read_csv("./data_trans/classMap.csv")
colnames(preds) <- paste("rf2",classMap$country_destination,sep="_")
sampID <- read_csv('sample_submission_NDF.csv')$id
sampID <- sort(sampID)
write.csv(data.frame(id=sampID, preds), "./stack_models/testPredsProbs_rf2.csv", row.names=FALSE)
colnames(preds) <- classMap$country_destination
testPreds_top5 <- as.vector(apply(preds, 1, function(x) names(sort(x)[12:8])))

# create submission 
idx = sampID
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/testPreds_rf2.csv", quote=FALSE, row.names = FALSE)









