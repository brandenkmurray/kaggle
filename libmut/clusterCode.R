library(caret)
library(doParallel)
library(xgboost)
library(caretEnsemble)
library(randomForest)
library(Matrix)
library(qdapTools)
library(e1071)
NormalizedGini <- function(data, lev = NULL, model = NULL) {
  SumModelGini <- function(solution, submission) {
    df = data.frame(solution = solution, submission = submission)
    df <- df[order(df$submission, decreasing = TRUE),]
    df$random = (1:nrow(df))/nrow(df)
    totalPos <- sum(df$solution)
    df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
    df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
    df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
    return(sum(df$Gini))
  }
  
  solution=data$obs
  submission=data$pred
  result=SumModelGini(solution, submission) / SumModelGini(solution, solution)
  names(result) <- "Gini"
  result
}


setwd("/home/branden/Documents/kaggle/libmut")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# keep copy of ID variables for test and train data
train_Id <- train$Id
test_Id <- test$Id
train_Hazard <- train$Hazard
# Drop Id variable
train$Id <- NULL
train$T2_V10 <- NULL
train$T2_V7 <- NULL
train$T1_V10 <- NULL
train$T1_V13 <- NULL
train$Hazard <- NULL


factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)
  
  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

vars <- sapply(train, is.factor)
# names(vars[vars]) returns column names of factor variables
addDimTrain <- factorToNumeric(train, train, "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))
addDimTest <- factorToNumeric(train, test, "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))

impVars <- c('T1_V1', 'T1_V2', 'T1_V3', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V8', 'T2_V9', 'T2_V14', 'T2_V15', 'mean_T1_V4', 'median_T1_V4', 'sd_T1_V4', 'skewness_T1_V4', 'kurtosis_T1_V4', 'mean_T1_V5', 'median_T1_V5', 'sd_T1_V5', 'skewness_T1_V5', 'kurtosis_T1_V5', 'mean_T1_V7', 'median_T1_V7', 'sd_T1_V7', 'mean_T1_V8', 'median_T1_V8', 'sd_T1_V8', 'mean_T1_V9', 'median_T1_V9', 'sd_T1_V9', 'skewness_T1_V9', 'kurtosis_T1_V9', 'mean_T1_V11', 'median_T1_V11', 'sd_T1_V11', 'skewness_T1_V11', 'kurtosis_T1_V11', 'mean_T1_V12', 'median_T1_V12', 'sd_T1_V12', 'skewness_T1_V12', 'kurtosis_T1_V12', 'mean_T1_V15', 'median_T1_V15', 'sd_T1_V15', 'skewness_T1_V15', 'kurtosis_T1_V15', 'mean_T1_V16', 'median_T1_V16', 'sd_T1_V16', 'skewness_T1_V16', 'kurtosis_T1_V16', 'mean_T1_V17', 'sd_T1_V17', 'skewness_T1_V17', 'kurtosis_T1_V17', 'mean_T2_V5', 'median_T2_V5', 'sd_T2_V5', 'skewness_T2_V5', 'kurtosis_T2_V5', 'mean_T2_V11', 'sd_T2_V11', 'skewness_T2_V11', 'kurtosis_T2_V11', 'mean_T2_V12', 'sd_T2_V12', 'skewness_T2_V12', 'kurtosis_T2_V12', 'mean_T2_V13', 'median_T2_V13', 'sd_T2_V13', 'skewness_T2_V13', 'kurtosis_T2_V13')
addDimTrain <- addDimTrain[,colnames(addDimTrain) %in% impVars]
addDimTest <- addDimTest[,colnames(addDimTest) %in% impVars]

train <- cbind(train, addDimTrain)
test <- cbind(test, addDimTest)

numCols <- sapply(train, is.numeric)
train <- train[, colnames(train) %in% names(which(numCols==TRUE))]
test <- test[, colnames(test) %in% names(which(numCols==TRUE))]


preProc <- preProcess(train, method=c("center","scale"))
trainProc <- predict(preProc, train)
testProc <- predict(preProc, test)

clustSet <- rbind(trainProc, testProc)

kMeansTrain1 <- kmeans(clustSet, centers=2, iter.max=1000)
trainClusters <- head(kMeansTrain1$cluster,nrow(trainProc))
testClusters <- tail(kMeansTrain1$cluster,nrow(testProc))

cluster1 <- subset(trainProc, trainClusters==1)
cluster2 <- subset(trainProc, trainClusters==2)
cluster3 <- subset(trainProc, trainClusters==3)
cluster4 <- subset(trainProc, trainClusters==4)
cluster5 <- subset(trainProc, trainClusters==5)
cluster1$Hazard <- train_Hazard[trainClusters==1]
cluster2$Hazard <- train_Hazard[trainClusters==2]
cluster3$Hazard <- train_Hazard[trainClusters==3]
cluster4$Hazard <- train_Hazard[trainClusters==4]
cluster5$Hazard <- train_Hazard[trainClusters==5]

testProc$Id <- test_Id
testClust1 <- subset(testProc, testClusters==1)
testClust2 <- subset(testProc, testClusters==2)
testClust3 <- subset(testProc, testClusters==3)
testClust4 <- subset(testProc, testClusters==4)
testClust5 <- subset(testProc, testClusters==5)


xgbCtrl <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE,
                        selectionFunction="best",
                        summaryFunction=NormalizedGini,
                        verboseIter=TRUE)
xgbGrid <- expand.grid(max_depth = c(3,4),
                       nrounds = c(1000,1200),
                       eta = c(.005,.01,.015))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
xgbTrain <- train(Hazard~.,
                  data=cluster2,
                  method="xgbTree",
                  metric="Gini",
                  trControl=xgbCtrl,
                  tuneGrid=xgbGrid,
                  min_child_weight=10,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  verbose=1)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "xgbTrain Model", "Done.")


xgbC1v2 <- xgbTrain
xgbC2v2 <- xgbTrain
xgbC3 <- xgbTrain
xgbC4 <- xgbTrain
xgbC5 <- xgbTrain

preds1 <- predict(xgbC1v2, testClust1)
preds2 <- predict(xgbC2v2, testClust2)
preds3 <- predict(xgbC3, testClust3)
preds4 <- predict(xgbC4, testClust4)
preds5 <- predict(xgbC5, testClust5)

submission <- data.frame(Id = c(testClust1$Id, testClust2$Id), Hazard=c(preds1, preds2))
write.csv(submission, "SubmissionClusterV2XGB-07-19-2015.csv", row.names=FALSE)
