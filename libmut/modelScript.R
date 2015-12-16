library(caret)
library(doParallel)
library(xgboost)
library(caretEnsemble)
library(randomForest)
library(Matrix)
library(qdapTools)
library(e1071)
#Gini code from kaggle
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

NormalizedGini1 <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}
setwd("/home/branden/Documents/kaggle/libmut")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# keep copy of ID variables for test and train data
train_Id <- train$Id
test_Id <- test$Id
# Drop Id variable
train$Id <- NULL
train$T2_V10 <- NULL
train$T2_V7 <- NULL
train$T1_V10 <- NULL
train$T1_V13 <- NULL

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


newDim <- factorToNumeric(train, train, "Hazard", c("T2_V11","T2_V13"), "mean")
vars <- sapply(train, is.factor)
# names(vars[vars]) returns column names of factor variables
addDimTrain <- factorToNumeric(train, train, "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))
addDimTest <- factorToNumeric(train, test, "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))

impVars <- c("T1_V1","T1_V2","T1_V3","T1_V4","T1_V5","T1_V6","T1_V7","T1_V8","T1_V9","T1_V11","T1_V12","T1_V14","T1_V15","T1_V16","T1_V17","T2_V1","T2_V2","T2_V3","T2_V4","T2_V5","T2_V6","T2_V8","T2_V9","T2_V11","T2_V12","T2_V13","T2_V14","T2_V15",'mean_T1_V4', 'median_T1_V4', 'sd_T1_V4', 'skewness_T1_V4', 'kurtosis_T1_V4', 'mean_T1_V5', 'median_T1_V5', 'sd_T1_V5', 'skewness_T1_V5', 'kurtosis_T1_V5', 'mean_T1_V7', 'median_T1_V7', 'sd_T1_V7', 'mean_T1_V8', 'median_T1_V8', 'sd_T1_V8', 'mean_T1_V9', 'median_T1_V9', 'sd_T1_V9', 'skewness_T1_V9', 'kurtosis_T1_V9', 'mean_T1_V11', 'median_T1_V11', 'sd_T1_V11', 'skewness_T1_V11', 'kurtosis_T1_V11', 'mean_T1_V12', 'median_T1_V12', 'sd_T1_V12', 'mean_T1_V15', 'median_T1_V15', 'sd_T1_V15', 'skewness_T1_V15', 'kurtosis_T1_V15', 'mean_T1_V16', 'median_T1_V16', 'sd_T1_V16', 'skewness_T1_V16', 'kurtosis_T1_V16', 'mean_T1_V17', 'mean_T2_V5', 'median_T2_V5', 'sd_T2_V5', 'skewness_T2_V5', 'kurtosis_T2_V5', 'mean_T2_V11',  'mean_T2_V12',  'mean_T2_V13', 'median_T2_V13', 'sd_T2_V13', 'skewness_T2_V13')
addDimTrain <- addDimTrain[,colnames(addDimTrain) %in% impVars]
addDimTest <- addDimTest[,colnames(addDimTest) %in% impVars]

train <- cbind(train, addDimTrain)
test <- cbind(test, addDimTest)

numCols <- sapply(train, is.numeric)
train <- train[, colnames(train) %in% names(which(numCols==TRUE))]
test <- test[, colnames(test) %in% names(which(numCols==TRUE))]
# response variable from training data
train_y <- train$Hazard

# predictor variables from training
train_x <- subset(train, select = -c(Id, Hazard))
train_x <- sparse.model.matrix(~., data = train_x)

# predictor variables from test
test_x <- subset(test, select = -c(Id))
test_x <- sparse.model.matrix(~., data = test_x)

# Set xgboost parameters
param <- list("objective" = "reg:linear",
              "eta" = 0.02,
              "min_child_weight" = 5,
              "subsample" = .8,
              "colsample_bytree" = .8,
              "scale_pos_weight" = 1.0,
              "max_depth" = 4)

# Using 5000 rows for early stopping. 
offset <- 5000
num_rounds <- 2000

# Set xgboost test and training and validation datasets
xgtest <- xgb.DMatrix(data = test_x)
xgtrain <- xgb.DMatrix(data = train_x[offset:nrow(train_x),], label= log(train_y[offset:nrow(train_x)]))
xgval <-  xgb.DMatrix(data = train_x[1:offset,], label= train_y[1:offset])

# setup watchlist to enable train and validation, validation must be first for early stopping
watchlist <- list(val=xgval, train=xgtrain)
# to train with watchlist, use xgb.train, which contains more advanced features

evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini1(as.numeric(labels),as.numeric(preds))
  return(list(metric = "Gini", value = err))
}

# this will use default evaluation metric = rmse which we want to minimise
bst1 <- xgb.train(params = param, data = xgtrain, feval=evalgini, nround=num_rounds, print.every.n = 20, watchlist=watchlist, early.stop.round = 80, maximize = TRUE)






rfCtrl <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE,
                        selectionFunction="best",
                        summaryFunction=NormalizedGini,
                       verboseIter=TRUE)
rfGrid <- expand.grid(mtry=c(7,13,17))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
rfTrain <- train(log(Hazard) ~.,
                 data=train,
                 method="rf",
                 metric="Gini",
                 ntree=1500,
                 trControl=rfCtrl,
                 tuneGrid=rfGrid,
                 verbose=TRUE,
                 nodesize=10,
                 sampsize=10000,
                 importance=TRUE)
stopCluster(cl)
(rfTime <- Sys.time() - tme)
pbPost("note","rfTrain","Done.")

imp <- varImp(rfTrain$finalModel)
str(imp)
impLim <- imp>5
impNames <- rownames(imp)[imp$Overall>5]
paste("'",as.character(impNames),"'",collapse=", ",sep="")

# "'T1_V1', 'T1_V2', 'T1_V3', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V8', 'T2_V9', 'T2_V14', 'T2_V15', 'mean_T1_V4', 'median_T1_V4', 'sd_T1_V4', 'skewness_T1_V4', 'kurtosis_T1_V4', 'mean_T1_V5', 'median_T1_V5', 'sd_T1_V5', 'skewness_T1_V5', 'kurtosis_T1_V5', 'mean_T1_V7', 'median_T1_V7', 'sd_T1_V7', 'mean_T1_V8', 'median_T1_V8', 'sd_T1_V8', 'mean_T1_V9', 'median_T1_V9', 'sd_T1_V9', 'skewness_T1_V9', 'kurtosis_T1_V9', 'mean_T1_V11', 'median_T1_V11', 'sd_T1_V11', 'skewness_T1_V11', 'kurtosis_T1_V11', 'mean_T1_V12', 'median_T1_V12', 'sd_T1_V12', 'skewness_T1_V12', 'kurtosis_T1_V12', 'mean_T1_V15', 'median_T1_V15', 'sd_T1_V15', 'skewness_T1_V15', 'kurtosis_T1_V15', 'mean_T1_V16', 'median_T1_V16', 'sd_T1_V16', 'skewness_T1_V16', 'kurtosis_T1_V16', 'mean_T1_V17', 'sd_T1_V17', 'skewness_T1_V17', 'kurtosis_T1_V17', 'mean_T2_V5', 'median_T2_V5', 'sd_T2_V5', 'skewness_T2_V5', 'kurtosis_T2_V5', 'mean_T2_V11', 'sd_T2_V11', 'skewness_T2_V11', 'kurtosis_T2_V11', 'mean_T2_V12', 'sd_T2_V12', 'skewness_T2_V12', 'kurtosis_T2_V12', 'mean_T2_V13', 'median_T2_V13', 'sd_T2_V13', 'skewness_T2_V13', 'kurtosis_T2_V13'"

impVars <- c('T1_V1', 'T1_V2', 'T1_V3', 'T1_V14', 'T2_V1', 'T2_V2', 'T2_V4', 'T2_V6', 'T2_V8', 'T2_V9', 'T2_V14', 'T2_V15', 'mean_T1_V4', 'median_T1_V4', 'sd_T1_V4', 'skewness_T1_V4', 'kurtosis_T1_V4', 'mean_T1_V5', 'median_T1_V5', 'sd_T1_V5', 'skewness_T1_V5', 'kurtosis_T1_V5', 'mean_T1_V7', 'median_T1_V7', 'sd_T1_V7', 'mean_T1_V8', 'median_T1_V8', 'sd_T1_V8', 'mean_T1_V9', 'median_T1_V9', 'sd_T1_V9', 'skewness_T1_V9', 'kurtosis_T1_V9', 'mean_T1_V11', 'median_T1_V11', 'sd_T1_V11', 'skewness_T1_V11', 'kurtosis_T1_V11', 'mean_T1_V12', 'median_T1_V12', 'sd_T1_V12', 'skewness_T1_V12', 'kurtosis_T1_V12', 'mean_T1_V15', 'median_T1_V15', 'sd_T1_V15', 'skewness_T1_V15', 'kurtosis_T1_V15', 'mean_T1_V16', 'median_T1_V16', 'sd_T1_V16', 'skewness_T1_V16', 'kurtosis_T1_V16', 'mean_T1_V17', 'sd_T1_V17', 'skewness_T1_V17', 'kurtosis_T1_V17', 'mean_T2_V5', 'median_T2_V5', 'sd_T2_V5', 'skewness_T2_V5', 'kurtosis_T2_V5', 'mean_T2_V11', 'sd_T2_V11', 'skewness_T2_V11', 'kurtosis_T2_V11', 'mean_T2_V12', 'sd_T2_V12', 'skewness_T2_V12', 'kurtosis_T2_V12', 'mean_T2_V13', 'median_T2_V13', 'sd_T2_V13', 'skewness_T2_V13', 'kurtosis_T2_V13')


### NEURAL NET 
nnetCtrl <- trainControl(method="cv",
                         number=5,
                         allowParallel=TRUE,
                         selectionFunction="best",
                         summaryFunction=NormalizedGini,
                         verboseIter=TRUE)

### M5
m5Ctrl <- trainControl(method="cv",
                         number=3,
                         allowParallel=TRUE,
                         selectionFunction="best",
                         summaryFunction=NormalizedGini,
                         verboseIter=TRUE)
m5Grid <- expand.grid(pruned=c("Yes"), smoothed=c("Yes"), rules=c("Yes"))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
m5Train <- train(Hazard ~ .,
                  data=train,
                  method="M5",
                  metric="Gini",
                  trControl=m5Ctrl,
                  tuneGrid=m5Grid)
stopCluster(cl)
Sys.time() - tme
pbPost("note","M5","Done.")
m5Train

### NNET
nnetCtrl <- trainControl(method="cv",
                       number=3,
                       allowParallel=TRUE,
                       selectionFunction="best",
                       # summaryFunction=NormalizedGini,
                       verboseIter=TRUE)
nnetGrid <- expand.grid(size=c(1,2,3,4), decay=c(0,0.001,0.01,0.1,0.2,0.3,0.4,0.5))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
nnetTrain <- train(Hazard ~ .,
                 data=train,
                 method="neuralnet",
                 # metric="Gini",
                 trControl=nnetCtrl,
                 # tuneGrid=nnetGrid)
                 tuneLength=1)
stopCluster(cl)
Sys.time() - tme
pbPost("note","NNET","Done.")
nnetTrain


### NNET
svmCtrl <- trainControl(method="cv",
                         number=3,
                         allowParallel=TRUE,
                         selectionFunction="best",
                         summaryFunction=NormalizedGini,
                         verboseIter=TRUE)
svmGrid <- expand.grid(sigma=c(.008,.01), C=c(.15,.2,.22))

trainTrans <- train
trainTrans$Hazard <- log(trainTrans$Hazard)


cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
svmTrain <- train(Hazard ~ .,
                   data=trainTrans,
                   method="svmRadial",
                   metric="Gini",
                   preProcess=c("center","scale"),
                   trControl=svmCtrl,
                   tuneGrid=svmGrid)
                   # tuneLength=1)
stopCluster(cl)
Sys.time() - tme
pbPost("note","SVM","Done.")
svmTrain

### GLMNET
glmnetCtrl <- trainControl(method="cv",
                         number=5,
                         allowParallel=TRUE,
                         selectionFunction="best",
                         summaryFunction=NormalizedGini,
                         verboseIter=TRUE)
glmnetGrid <- expand.grid(alpha=c())

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
glmnetTrain <- train(Hazard ~ .,
                   data=train,
                   method="glmnet",
                   metric="Gini",
                   trControl=glmnetCtrl,
                   # tuneGrid=nnetGrid)
                   tuneLength=2)
stopCluster(cl)
Sys.time() - tme
pbPost("note","NNET","Done.")
glmnetTrain


## XGBOOST LINEAR
xgbCtrl <- trainControl(method="cv",
                       number=5,
                       allowParallel=TRUE,
                       selectionFunction="best",
                       summaryFunction=NormalizedGini,
                       verboseIter=TRUE)
xgbGrid <- expand.grid(nrounds=floor((1:3) * 15),
                       lambda = c(0, 10^seq(-1, -4, length = 2 - 1)), 
                       alpha = c(0, 10^seq(-.5, -1, length = 3 - 1)))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
xgbTrain <- train(Hazard ~ .,
                 data=train,
                 method="xgbLinear",
                 metric="Gini",
                 trControl=xgbCtrl,
                 tuneGrid=xgbGrid,
                 verbose=1)
stopCluster(cl)
Sys.time() - tme

## KNN
knnCtrl <- trainControl(method="cv",
                                   number=5,
                                   allowParallel=TRUE,
                                   selectionFunction="best",
                                   summaryFunction=NormalizedGini,
                                   verboseIter=TRUE)
knnGrid <- expand.grid(k=c(100,150,200))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
knnTrain <- train(Hazard~.,
                  data=train,
                  method="knn",
                  metric="Gini",
                  trControl=knnCtrl,
                  tuneGrid=knnGrid,
                  preProcess=c("center","scale")
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "knnTrain Model", "Done.")


## XGBOOST TREES
xgbCtrl <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE,
                        selectionFunction="best",
                        summaryFunction=NormalizedGini,
                        verboseIter=TRUE)
xgbGrid <- expand.grid(max_depth = c(9),
                       nrounds = c(1000),
                       eta = c(.005))

train1 <- train
train1$Hazard <- train1$Hazard^2

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
xgbTrain <- train(Hazard~.,
                  data=train,
                  method="xgbTree",
                  metric="Gini",
                  trControl=xgbCtrl,
                  tuneGrid=xgbGrid,
                  objective="reg:linear",
                  min_child_weight=6,
                  subsample=0.7,
                  colsample_bytree=0.7,
                  verbose=1)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "xgbTrain Model", "Done.")

pred <- predict(xgbTrain, test)
submission = data.frame(Id = test_Id, Hazard = pred)
write.csv(submission, "SubmissionXGBoostHazardSquared-07-31-2015.csv", row.names=FALSE)



pred1 <- predict(rfTrain, test)
submission = data.frame(Id = test$Id, Hazard = pred)
write.csv(submission, "SubmissionRF-07-10-2015.csv", row.names=FALSE)


## GBM
gbmCtrl <- trainControl(method="cv",
                        number=5,
                        allowParallel=TRUE,
                        selectionFunction="best",
                        summaryFunction=NormalizedGini,
                        verboseIter=TRUE)
gbmGrid <- expand.grid(interaction.depth=c(7,13,19),
                       n.trees=c(2000),
                       shrinkage=c(.005,.01,.015),
                       n.minobsinnode=c(1,5))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
gbmTrain <- train(Hazard~.,
                  data=train,
                  method="gbm",
                  metric="Gini",
                  trControl=gbmCtrl,
                  tuneGrid=gbmGrid)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "gbmTrain Model", "Done.")


pred2 <- predict(gbmTrain, test)
head(preds1)
head(preds2)

predsComb <- 5*pred1 + 2*pred2
submission = data.frame(Id = test_Id, Hazard = predsComb)
write.csv(submission, "SubmissionXGB+GBMArbitraryCombo-07-18-2015.csv", row.names=FALSE)


## Caret Ensemble
ensCtrl <- trainControl(method="cv",
                        number=5,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createMultiFolds(train$Hazard, k=5, times=5),
                        selectionFunction="best",
                        summaryFunction=NormalizedGini)
xgbGrid <- expand.grid(max_depth = c(5),
                       nrounds = c(1200),
                       eta = c(.015))
gbmGrid <- expand.grid(interaction.depth=c(19),
                       n.trees=c(3000),
                       shrinkage=c(.01),
                       n.minobsinnode=c(20))
rfGrid <- expand.grid(mtry=c(13))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
model_list <- caretList(
  log(Hazard) ~ .,
  data=train,
  trControl=ensCtrl,
  metric="Gini",
  tuneList=list(
    rf1=caretModelSpec(method="rf", 
                      tuneGrid=expand.grid(mtry=c(13)), 
                      nodesize=10, 
                      ntree=1500),
    rf2=caretModelSpec(method="rf", 
                      tuneGrid=expand.grid(mtry=c(17)), 
                      nodesize=10, 
                      ntree=1500),
    xgb1=caretModelSpec(method="xgbTree", 
                       tuneGrid=expand.grid(max_depth = c(5),
                                            nrounds = c(1200),
                                            eta = c(.015)),
                       preProcess=c("center","scale"),
                       min_child_weight=10,
                       subsample=0.8,
                       colsample_bytree=0.8),
    xgb2=caretModelSpec(method="xgbTree", 
                       tuneGrid=expand.grid(max_depth = c(7),
                                            nrounds = c(1200),
                                            eta = c(.015)),
                       preProcess=c("center","scale"),
                       min_child_weight=10,
                       subsample=0.8,
                       colsample_bytree=0.8),
    xgb3=caretModelSpec(method="xgbTree", 
                       tuneGrid=expand.grid(max_depth = c(3),
                                            nrounds = c(1200),
                                            eta = c(.015)),
                       preProcess=c("center","scale"),
                       min_child_weight=5,
                       subsample=0.7,
                       colsample_bytree=0.7),
    gbm1=caretModelSpec(method="gbm", 
                       preProcess=c("center","scale"),
                       tuneGrid=expand.grid(interaction.depth=c(19),
                                            n.trees=c(3000),
                                            shrinkage=c(.01),
                                            n.minobsinnode=c(20))),
    gbm1=caretModelSpec(method="gbm", 
                        preProcess=c("center","scale"),
                        tuneGrid=expand.grid(interaction.depth=c(13),
                                             n.trees=c(3000),
                                             shrinkage=c(.01),
                                             n.minobsinnode=c(5))))
  )
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "Ensemble", "Finished.")

save(model_list, file="model_list-RF-XGB-GBM.rda")

xyplot(resamples(model_list))
modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)

library('caTools')
model_preds <- lapply(model_list, predict, newdata=ts1[split1==2,])
#model_preds <- lapply(model_preds, function(x) x[,'Yes'])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=ts1[ts1$split1==2,])
model_preds$ensemble <- ens_preds

ens_sum_preds <- rowSums(model_preds)

submission = data.frame(Id = ts1$Id[ts1$split1==2], Hazard = ens_preds)
write.csv(submission, "SubmissionEnsemble-rf-xgb-gbm-08012015.csv", row.names=FALSE)

# Not tuning stack
models <- caretList(rf=rfTrain, gbm=gbmTrain)
caretEns <- caretEnsemble(models)

# Ensemble Stack
cl <- makeCluster(7)
registerDoParallel(cl)
tme <- Sys.time()
gbm_stack <- caretStack(
  model_list, 
  method='gbm',
  verbose=FALSE,
  tuneGrid=expand.grid(n.trees=c(2000), interaction.depth=c(17,21,25), shrinkage=c(.01,.001), n.minobsinnode=c(5,10)),
  metric='Gini',
  trControl=trainControl(
    method='cv',
    number=5,
    savePredictions=FALSE,
    allowParallel=TRUE,
    summaryFunction=NormalizedGini
  )
)
stopCluster(cl)
Sys.time() - tme
gbm_stack
pbPost("note","Stack","Done.")

gbm.perf(gbm_stack$ens_model$finalModel, oobag.curve=TRUE)

model_preds2 <- model_preds
stackPred <- predict(gbm_stack, newdata=test, type='raw')
#CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
#colAUC(model_preds2, testing$Class)


gbmStackTest <- predict(gbm_stack, newdata=testing, type="prob", na.action=na.pass)$Yes
gbmTable <- data.frame(bidder_id=testingID, gbmStackTest)
gbmTableMerge <- merge(tail(allMerge, nrow(test)), gbmTable, by="bidder_id", all.x=TRUE)

submission = data.frame(Id = test_Id, Hazard = stackPred)



## Creat Submission File
write.csv(submission, "SubmissionGBMStack-xgb-rf-gbm-08032015.csv", row.names=FALSE)



library(gamlss)

gamMod <- gamlss(Hazard~., data=train, family=NBI())
summary(gamMod)
Sys.time()
gamPred <- predict(gamMod, newdata=test, type="response")
hist(gamPred, breaks=breaks)
hist(stackPred, breaks=breaks)


## Ensemble small
ensCtrl <- trainControl(method="cv",
                        number=5,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createMultiFolds(train$Hazard, k=5, times=5),
                        selectionFunction="best",
                        summaryFunction=NormalizedGini)
xgbGrid <- expand.grid(max_depth = c(5),
                       nrounds = c(1200),
                       eta = c(.015))
gbmGrid <- expand.grid(interaction.depth=c(19),
                       n.trees=c(3000),
                       shrinkage=c(.01),
                       n.minobsinnode=c(20))
rfGrid <- expand.grid(mtry=c(13))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
model_list <- caretList(
  log(Hazard) ~ .,
  data=train,
  trControl=ensCtrl,
  metric="Gini",
  tuneList=list(
    rf2=caretModelSpec(method="rf", 
                       tuneGrid=expand.grid(mtry=c(17)), 
                       nodesize=10, 
                       ntree=1500,
                       sampsize=10000),
    xgb1=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(5),
                                             nrounds = c(1200),
                                             eta = c(.015)),
                        preProcess=c("center","scale"),
                        min_child_weight=10,
                        subsample=0.8,
                        colsample_bytree=0.8)
  )
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "Ensemble", "Finished.")

save(model_list, file="model_list-RF-XGB-GBM.rda")

xyplot(resamples(model_list))
modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)

library('caTools')
model_preds <- lapply(model_list, predict, newdata=test)
#model_preds <- lapply(model_preds, function(x) x[,'Yes'])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=test)
model_preds$ensemble <- ens_preds

ens_sum_preds <- rowSums(model_preds)

submission = data.frame(Id = test_Id, Hazard = ens_preds)
write.csv(submission, "SubmissionEnsemble-rf-xgb-08032015.csv", row.names=FALSE)


