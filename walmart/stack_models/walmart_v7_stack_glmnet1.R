library(glmnet)
library(readr)
library(data.table)
library(caret)
library(gtools)
setwd("/home/branden/Documents/kaggle/walmart")
ts1Trans <- data.table(read_csv("./stack_models/ts1Trans3_abs.csv"))

# Create a vector the variable names to be used
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))

# Was only necessary for easier filtering of the validation set
train <- ts1Trans[filter==0, c("class",varnames), with=FALSE]
test <- ts1Trans[filter==2, c("class",varnames), with=FALSE]
# Convert level X8 to most common label (X5) -- not enough observations -- just predict all 0s
train$class[train$class==8] <- 5
train$class <- as.factor(make.names(train$class))
# 
# library(doParallel)
# cl <- makeCluster(14)
# registerDoParallel(cl)
# nzv_fine <- nearZeroVar(train[,115:5283,with=FALSE], freqCut= 999, uniqueCut= 5, foreach=TRUE, allowParallel = TRUE)
# stopCluster(cl)

pca_fine <- preProcess(train[,115:5283,with=FALSE], method=c("BoxCox","pca","center","scale"), pcaComp=200)
train_pca_fine <- predict(pca_fine, train[,115:5283,with=FALSE])
test_pca_fine <- predict(pca_fine, newdata=test[,115:5283,with=FALSE])

pp <- preProcess(train[,3:114, with=FALSE], method=c("BoxCox","center","scale"))
train_pp <- predict(pp, newdata=train[,3:114,with=FALSE])
test_pp <- predict(pp, newdata=test[,3:114,with=FALSE])

train_conc <- cbind(train_pp, train_pca_fine)
test_conc <- cbind(test_pp, test_pca_fine)

# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
load("./stack_models/cvFoldsList.rda")


# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}


glmnetControl <- trainControl(method="cv",
                              number=5,
                              summaryFunction=mnLogLoss,
                              savePredictions=TRUE,
                              classProbs=TRUE,
                              index=cvFolds,
                              allowParallel=TRUE)
glmnetGrid <- expand.grid(alpha=c(.1,.2,.3), lambda=c(.00001,.0001))

library(doParallel)
cl <- makeCluster(14)
registerDoParallel(cl)
set.seed(2015)
glmnet1 <- train(x=train_conc,
                 y=train$class,
                 method="glmnet",
                 trControl=glmnetControl,
                 tuneGrid=glmnetGrid,
                 metric="logLoss",
                 tuneLength=1)
stopCluster(cl)
save(glmnet1, file="./stack_models/glmnet.rda")

cvPreds <- glmnet1$pred[glmnet1$pred$alpha==glmnet1$bestTune$alpha & glmnet1$pred$lambda==glmnet1$bestTune$lambda,3:40]
cvPreds$X8 <- 0
cvPreds <- cvPreds[order(cvPreds$rowIndex),mixedorder(names(cvPreds))]
cvPreds$rowIndex <- NULL

samp <- read.csv('sample_submission.csv')
cnames <- paste("glmnet1", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(VisitNumber=ts1Trans[filter==0,"VisitNumber",with=FALSE],cvPreds), "./stack_models/cvPreds_glmnet1.csv", row.names=FALSE) 


preds <- predict(glmnet1, newdata=test_conc, type="prob")
preds$X8 <- 0
preds <- preds[,mixedorder(names(preds))]
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/testPreds_glmnet1full.csv", row.names=FALSE)















glmnet1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(train), ncol=38)) 
colnames(glmnet1_stack_preds) <- unique(train$class)
glmnet1_stack_preds <- glmnet1_stack_preds[,mixedorder(colnames(glmnet1_stack_preds))]
logLossTable <- data.frame(fold=seq(1:length(cvFolds)), LogLoss=rep(0, length(cvFolds)))
(tme <- Sys.time())
for (i in 1:length(cvFolds)){
  set.seed(201510)
  glmnet1 <- glmnet(x=data.matrix(train_pp[cvFolds$x!=1,1:113,with=F]),
                    y=train$class[cvFolds$x!=1],
                    family="multinomial",
                    alpha = 0,
                    nlambda = 100
                    
  )
  glmnet1_pred <- predict(glmnet1,  newx=data.matrix(train_pp[cvFolds$x==1,1:113, with=FALSE]), type="response")
  glmnet1_pred <- as.data.frame(glmnet1_pred)
  glmnet1_pred <- glmnet1_pred[,mixedorder(names(glmnet1_pred))]
  glmnet1_stack_preds[cvFolds!=1,colnames(tmp)] <- tmp
  LogLoss(model.matrix(~as.factor(class)-1, train[cvFolds$x==1,"class", with=FALSE]), glmnet1_pred)
  #   
  # glmnet1_stack_preds[-cvFolds[[i]],] <- predict(glmnet1, newdata=data.matrix(train[-cvFolds[[i]],varnames, with=FALSE]), probability=TRUE)
  tmp <- as.data.frame(predict(glmnet1, newdata=data.matrix(train[-cvFolds[[i]],varnames, with=FALSE]), probability=TRUE))
  tmp <- tmp[,mixedorder(colnames(tmp))]
  glmnet1_stack_preds[-cvFolds[[i]],colnames(tmp)] <- tmp
  actTmp <- as.data.frame(model.matrix(~as.factor(class)-1, train[-cvFolds[[i]],"class", with=FALSE]))
  actTmp <- actTmp[,mixedorder(colnames(actTmp))]
  logLossTable[i,2] <- LogLoss(actTmp, glmnet1_stack_preds[-cvFolds[[i]],])
}
Sys.time() - tme
