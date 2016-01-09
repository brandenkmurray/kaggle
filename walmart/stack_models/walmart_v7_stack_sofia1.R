library(RSofia)
library(readr)
library(data.table)
library(caret)
library(gtools)
setwd("/home/branden/Documents/kaggle/walmart")
ts1Trans <- data.table(read_csv("./data_trans/ts1Trans4_prop_simil.csv", col_types=paste(replicate(8280, "n"), collapse = "")))

# Create a vector the variable names to be used
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))

# Was only necessary for easier filtering of the validation set
train <- ts1Trans[filter==0, c("class",varnames), with=FALSE]
test <- ts1Trans[filter==2, c("class",varnames), with=FALSE]
# Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
# train$class[train$class==8] <- 5
# train$class <- as.factor(make.names(train$class))
# 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)
# train$class <- as.factor(make.names(train$class))
# classDummy <- dummyVars(~ class , train, levelsOnly = TRUE)
# classDummyMat <- predict(classDummy, train)
# classDummyMat <- classDummyMat[,mixedorder(colnames(classDummyMat))]

pp <- preProcess(train[,2:ncol(train),with=FALSE], method=c("BoxCox","center","scale"))
train_pp <- predict(pp, train[,2:ncol(train),with=FALSE])
test_pp <- predict(pp, newdata=test[,2:ncol(test),with=FALSE])

train_pp <- cbind(class=train$class, train_pp)

# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
load("./data_trans/cvFoldsList.rda")

# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}


sofia1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(train), ncol=38)) 
colnames(sofia1_stack_preds) <- colnames(classDummyMat)
logLossTable <- data.frame(fold=seq(1:length(cvFoldsList)), LogLoss=rep(0, length(cvFoldsList)))
(tme <- Sys.time())
for (fold in 1:length(cvFoldsList)){
  for (i in 1:38){
    print(paste0("fold ",fold," class ",i))
    set.seed(201510)
    train_pp$class2 <- as.numeric(train_pp$class == (i-1))
    sofia1 <- sofia(class2 ~ .  , 
                    data=train_pp[-cvFoldsList[[fold]], c("class2",varnames), with=FALSE],
                      random_seed=76,
                      learner_type="logreg-pegasos",
                      loop_type = "balanced-stochastic"
    )
    # sofia1_pred <- predict(sofia1, newdata=train_pp[cvFoldsList[[1]],c("class2",varnames), with=FALSE], prediction_type="logistic")
    # LogLoss(model.matrix(~as.factor(class)-1, train[-cvFoldsList[[1]],"class", with=FALSE]), sofia1_pred)
    #   
    # sofia1_stack_preds[-cvFoldsList[[i]],] <- predict(sofia1, newdata=data.matrix(train[-cvFoldsList[[i]],varnames, with=FALSE]), probability=TRUE)
#     tmp <- as.data.frame(predict(sofia1, newdata=data.matrix(train[cvFoldsList[[i]],varnames, with=FALSE]), prediction_type="logistic"))
#     tmp <- tmp[,mixedorder(colnames(tmp))]
    sofia1_stack_preds[cvFoldsList[[fold]], i] <- predict(sofia1, newdata=train_pp[cvFoldsList[[fold]],c("class2",varnames), with=FALSE], prediction_type="logistic")
  } 
    actTmp <- as.data.frame(model.matrix(~as.factor(class)-1, train_pp[cvFoldsList[[fold]],"class", with=FALSE]))
    actTmp <- actTmp[,mixedorder(colnames(actTmp))]
    logLossTable[i,2] <- LogLoss(actTmp, sofia1_stack_preds[cvFoldsList[[fold]],])
}
Sys.time() - tme
logLossTable



for (i in 1:38){
  print(paste0("fold ",fold," class ",i))
  set.seed(201510)
  train_pp$class2 <- as.numeric(train_pp$class == (i-1))
  sofia1 <- sofia(class2 ~ .  , 
                  data=train_pp, c("class2",varnames), with=FALSE],
                  random_seed=76,
                  learner_type="logreg-pegasos",
                  loop_type = "balanced-stochastic"
  )
  # sofia1_pred <- predict(sofia1, newdata=train_pp[cvFoldsList[[1]],c("class2",varnames), with=FALSE], prediction_type="logistic")
  # LogLoss(model.matrix(~as.factor(class)-1, train[-cvFoldsList[[1]],"class", with=FALSE]), sofia1_pred)
  #   
  # sofia1_stack_preds[-cvFoldsList[[i]],] <- predict(sofia1, newdata=data.matrix(train[-cvFoldsList[[i]],varnames, with=FALSE]), probability=TRUE)
  #     tmp <- as.data.frame(predict(sofia1, newdata=data.matrix(train[cvFoldsList[[i]],varnames, with=FALSE]), prediction_type="logistic"))
  #     tmp <- tmp[,mixedorder(colnames(tmp))]
  test_pp$class2 <- as.numeric(test_pp$class == (i-1))
  sofia1_stack_preds[, i] <- predict(sofia1, newdata=test_pp[,c("class2",varnames), with=FALSE], prediction_type="logistic")
} 
