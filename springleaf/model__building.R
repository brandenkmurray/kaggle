library(caret)
library(doParallel)
library(pROC)
library(RPushbullet)
set.seed(6488)
folds <- createFolds(trainInd$y, k=3, list=TRUE, returnTrain=TRUE)
xgbCtrl <- trainControl(method="none",
                        # number=3,
                        classProbs=TRUE,
                        summaryFunction=twoClassSummary
#                         index=folds,
#                         allowParallel=TRUE
  )
xgbGrid <- expand.grid(max_depth = c(16),
                       nrounds = c(15000),
                       eta = c(0.001))



varnames <- names(ts1[,grepl("VAR", names(ts1))])
diffnames <- names(trainInd[,grepl("daydiff", names(trainInd))])
# numnames2 <- names(ts1[,sapply(ts1, is.numeric)])
# 
# # take 700 most important features from previous xgb tune
# # xgbImpNames <- xgb1Imp$Feature[1:700]
# expVars <- names(t1[,grepl("exp", names(t1))])
xgb3NotImportant <- xgb3Imp$Feature[1001:ncol(xgb3Imp)]
xgb3Important <- xgb3Imp$Feature[1:1000]
# trainInd <- trainInd[!names(trainInd) %in% xgb3NotImportant]
# testInd <- testInd[!names(testInd) %in% xgb3NotImportant]
xgb6Imp <- read.csv("xgb6Imp.csv")
xgb6NotImportant <- xgb6Imp$Feature[801:ncol(xgb6Imp)]
xgb6Important <- xgb6Imp$Feature[1:800]


cl <- makeCluster(4)
registerDoParallel(cl)
tme <- Sys.time()
(xgb7 <- train(y ~ ., 
               data=trainInd[,c("y",paste0(xgb6Important))],
               method="xgbTree",
               metric="ROC",
               trControl=xgbCtrl,
               tuneGrid=xgbGrid,
               objective="binary:logistic",
               min_child_weight=1,
               subsample=0.8,
               colsample_bytree=0.7,
               verbose=1)
)
(xgbTime <- Sys.time() - tme)
stopCluster(cl)
pbPost(type = "note", title = "XGB", body="Done.")
save(xgb7, file="xgb7.rda")

## Use a subset to determine important variables? Then build model using them
xgb6Imp <- xgb.importance(feature_names = xgb6$finalModel$xNames, model=xgb6$finalModel)
write.csv(xgb6Imp,"xgb6Imp.csv")
View(xgb6Imp)
xgb3Imp <- read.csv("xgb3Imp.csv", stringsAsFactors=FALSE)

library(data.table)
testInd <- fread("testInd-09-14-2015.csv")
targetPred <- predict(xgb7, testInd, type="prob")[2]
submission <- data.frame(ID=testId, target=targetPred)
colnames(submission)[2] <- "target"
write.csv(submission, "submission-09-19-2015-xgb7-xgb6Imp.csv", row.names=FALSE)

rfPred <- predict(rf1, testInd, type="prob")[2]
rfsubmission <- data.frame(ID=testId, target=rfPred)
colnames(rfsubmission)[2] <- "target"
write.csv(rfsubmission, "submission-09-18-2015-rf1Preds-nosubmit.csv", row.names=FALSE)

xgb3Important <- xgb3Imp$Feature[1:1000]
xgb3NotImportant <- xgb3Imp$Feature[1001:ncol(xgb3Imp)]
head(xgb3Important)

newPred <- .9*targetPred + .1*rfPred
blendsubmission <- data.frame(ID=testId, target=newPred)
colnames(blendsubmission)[2] <- "target"
write.csv(blendsubmission, "submission-09-18-2015-xgb6-rf1-blend-90-10.csv", row.names=FALSE)


###############################################################################
## GLMNET
glmnetCtrl <- trainControl(method="cv",
                        number=3,
                        classProbs=TRUE,
                        summaryFunction=twoClassSummary,
                        allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.001), lambda=c(.01))



cl <- makeCluster(3)
registerDoParallel(cl)
tme <- Sys.time()
(glmnet1 <- train(y ~ ., 
               data=trainInd,
               method="glmnet",
               metric="ROC",
               trControl=glmnetCtrl,
               tuneGrid=glmnetGrid)
)
(glmnetTime <- Sys.time() - tme)
stopCluster(cl)


###############################################################################
## RANDOM FOREST
rfCtrl <- trainControl(method="cv",
                           number=3,
                           classProbs=TRUE,
                           summaryFunction=twoClassSummary,
                           allowParallel=TRUE)
rfGrid <- expand.grid(mtry=c(15))



cl <- makeCluster(3)
registerDoParallel(cl)
tme <- Sys.time()
(rf1 <- train(y ~ ., 
                  data=trainInd,
                  method="rf",
                  metric="ROC",
                  ntree=300,
                  nodesize=1,
                  trControl=rfCtrl,
                  tuneGrid=rfGrid)
)
(rfTime <- Sys.time() - tme)
stopCluster(cl)
save(rf1, file="rf1.rda")



rfImp <- as.data.frame(importance(rf1$finalModel))
head(rfImp[sort(rfImp$MeanDecreaseGini),])
