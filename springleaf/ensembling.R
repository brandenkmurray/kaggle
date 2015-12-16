library(caretEnsemble)
library(xgboost)
library(pROC)
library(RPushbullet)
setwd("/home/branden/Documents/kaggle/springleaf")
# load("xgb13.rda")

varnames <- names(train[,grepl("VAR", names(train))])
# xgb13Imp <- xgb.importance(feature_names = NULL, model= xgb13)
# write.csv(xgb13Imp, "xgb13Imp.csv")


## TESTING CARET MODELS BEFORE ENSEMBLING
glmnetCtrl <- trainControl(method="cv",
                          number=2,
                        
                          classProbs = TRUE,
                          # allowParallel=TRUE,
                          # index=createMultiFolds(ts1$Hazard[ts1$split==0], k=2, times=2),
                          selectionFunction="best",
                          summaryFunction=twoClassSummary)

varnamessub <- names(train[2:50])


set.seed(2015)
ensCtrl <- trainControl(method="cv",
                        number=2,
                        classProbs=TRUE,
                        savePredictions=TRUE,
                        # allowParallel=TRUE,
                        index=createMultiFolds(train$target, k=2, times=2),
                        selectionFunction="best",
                        summaryFunction=twoClassSummary)
glmnet <- train(#x=train[,varnames],
  #y=factor(make.names(train$target)),
  
  factor(make.names(target))  ~ .,
  data=train[,c(varnamessub, "target")],
  method="glmnet",
  trControl=ensCtrl,
  metric="ROC",
  preProcess=c("center","scale"),
  tuneGrid=expand.grid(alpha=c(.1,.2), lambda=c(.05,.2)))

rf1 <- train(#x=train[,varnames],
  #y=factor(make.names(train$target)),
  
  factor(make.names(target))  ~ .,
  data=train[,c(varnamessub, "target")],
  method="rf",
  trControl=ensCtrl,
  metric="ROC",
  tuneGrid=expand.grid(mtry=c(5)), 
  nodesize=20, 
  ntree=20)

tme <- Sys.time()
xgbEns <- train(factor(make.names(target))  ~ .,
                data=train[,c(varnames, "target")],
                method="xgbTree", 
                metric="ROC",
                tuneGrid=expand.grid(max_depth = c(16),
                                     nrounds = c(20000),
                                     eta = c(.005)),
                min_child_weight=1,
                subsample=1,
                colsample_bytree=1)
(Sys.time() - tme)
save(xgbEns, file="xgbEns.rda")

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
model_list <- caretList(
  factor(make.names(target)) ~ .,
  data=train[,c("target",varnames)],
#   x=train[,2:50],
#   y=factor(make.names(train$target)),
  trControl=ensCtrl,
  metric="ROC",
  tuneList=list(
    #XGB tuned to that it creates a random forest model
    rf2=caretModelSpec(method="rf", 
                       tuneGrid=expand.grid(mtry=c(17)), 
                       nodesize=20, 
                       ntree=2000),
    xgb1=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(16),
                                             nrounds = c(20000),
                                             eta = c(.005)),
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1)
    
  )
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "Ensemble", "Finished.")


save(model_list, file="model_list-RF-XGB-GBM-GLMNET-SVM-08-21-2015.rda")


model_list <- list(glmnet=glmnet, rf1=rf1)
class(model_list) <- "caretList"
xyplot(resamples(model_list))
modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)


glm_ensemble <- caretStack(
  model_list, 
  method='glm',
  metric='ROC',
  trControl=trainControl(
    method='boot',
    number=10,
    savePredictions=TRUE,
    classProbs=TRUE,
    summaryFunction=twoClassSummary
  )
)
model_preds2 <- model_preds
model_preds2$ensemble <- predict(glm_ensemble, newdata=testing, type='prob')$M
CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds2, testing$Class)
