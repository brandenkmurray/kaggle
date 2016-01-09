library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
setwd("/home/branden/Documents/kaggle/walmart")

t1 <- data.table(read.csv("train.csv"))
s1 <- data.table(read.csv("test.csv"))

tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1$TripType <- NULL

s1$class <- -1
t1$filter <- 0
s1$filter <- 2

# list of top UPCs
topUPC <- names(sort(table(t1$Upc), decreasing=TRUE))[1:500]
t1$Upc2 <- ifelse(t1$Upc %in% topUPC, t1$Upc, "Other")
s1$Upc2 <- ifelse(s1$Upc %in% topUPC, s1$Upc, "Other")

ts1 <- rbind(t1, s1)
ts1[is.na(ts1)] <- -99

entropy <- function(x) {
  tab <- table(as.character(x))
  e <- sum(log(seq(1,sum(tab))))
  for (i in tab){
    e <- e - sum(log(seq(1,i)))
  }
  return(e)
}

entropy2 <- function(x, count) {
  tmp <- data.frame(x=x, count=count)
  tmp <- tmp[tmp$count>0,]
  if (nrow(tmp)==0)
    {return(-1)}
  else {
    tab <- aggregate(count ~ x, tmp, sum)
    e <- sum(log(seq(1,sum(tab$count))))
    for (i in tab$count){
      e <- e - sum(log(seq(1,i)))
    }
    return(e)
  }
}


data_transform <- function(data){
  data$ScanCount <- as.numeric(data$ScanCount)
  data$VisitNumber <- as.factor(data$VisitNumber)
  data$FinelineNumber <- as.factor(data$FinelineNumber)
  x <- data[, list(n=length(DepartmentDescription),
                   uniqDept=length(unique(DepartmentDescription)),
                   uniqFine=length(unique(FinelineNumber)),
                   uniqUpc=length(unique(Upc)),
                   deptEntropy=entropy(DepartmentDescription),
                   fineEntropy=entropy(FinelineNumber),
                   upcEntropy=entropy(Upc),
                   deptEntropy2=entropy2(DepartmentDescription, ScanCount),
                   fineEntropy2=entropy2(FinelineNumber, ScanCount),
                   upcEntropy2=entropy2(Upc, ScanCount),
                   purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
                   returns = -sum(ifelse(ScanCount<0,ScanCount,0)),
                   netScans=sum(abs(ScanCount)),
                   maxScans=max(ScanCount),
                   minScans=min(ScanCount),
                   meanScans=mean(ScanCount)), by=list(VisitNumber,class,filter)]
  x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                      upcDeptRatio=uniqUpc/uniqDept,
                      upcFineRatio=uniqUpc/uniqFine,
                      returnRatio = returns / netScans,
                      deptFineEntRatio=deptEntropy/fineEntropy,
                      deptUpcEntRatio=deptEntropy/upcEntropy,
                      fineUpcEntRatio=fineEntropy/upcEntropy,
                      deptFineEntRatio2=deptEntropy2/fineEntropy2,
                      deptUpcEntRatio2=deptEntropy2/upcEntropy2,
                      fineUpcEntRatio2=fineEntropy2/upcEntropy2,
                      scansDeptRatio=netScans/uniqDept,
                      scansFineRatio=netScans/uniqFine,
                      scansUpcRatio=netScans/uniqUpc)]
    
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="ScanCount",
                   fun.aggregate = sum, data=data)
  xWeekday <- data.table(xWeekday[,"VisitNumber",with=FALSE], prop.table(as.matrix(xWeekday[,2:ncol(xWeekday), with=FALSE]),margin=1))
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="ScanCount",
               fun.aggregate = sum, data=data)
  xDept <- data.table(xDept[,"VisitNumber",with=FALSE], prop.table(as.matrix(xDept[,2:ncol(xDept), with=FALSE]),margin=1))
  xFine <- dcast.data.table(VisitNumber~FinelineNumber, value.var="ScanCount",
                 fun.aggregate = sum, data=data)
  xFine <- data.table(xFine[,"VisitNumber",with=FALSE], prop.table(as.matrix(xFine[,2:ncol(xFine), with=FALSE]),margin=1))
  xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="ScanCount",
                            fun.aggregate = sum, data=data)
  xUpc <- data.table(xUpc[,"VisitNumber",with=FALSE], prop.table(as.matrix(xUpc[,2:ncol(xUpc), with=FALSE]),margin=1))
 
  xAgg <- merge(x, xWeekday, by="VisitNumber")
  xAgg <- merge(xAgg, xDept, by="VisitNumber")
  xAgg <- merge(xAgg, xFine, by="VisitNumber")
  xAgg <- merge(xAgg, xUpc, by="VisitNumber")
  return(xAgg)
}

ts1Trans <- data_transform(ts1)
# Reorder the data set so that the train set is at the top and then order by VisitNumber
ts1Trans <- ts1Trans[order(filter, VisitNumber),]
# Some entropy values were 0. This created some NAs for the entropy ratios
ts1Trans[is.na(ts1Trans)] <- 0

# Create a vector the variable names to be used
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))
# Create a separate train set -- This isn't necessary because I'm doing CV now instead of a single validation set 
# Was only necessary for easier filtering of the validation set
train <- ts1Trans[filter==0, c("class",varnames), with=FALSE]

# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
set.seed(2016)
cvFolds <- createFolds(train$class, k=5, list=TRUE, returnTrain=FALSE)
# dval<-xgb.DMatrix(data=data.matrix(train[h,varnames, with=FALSE]),label=data.matrix(train$class[h]))
dtrain<-xgb.DMatrix(data=data.matrix(train[,varnames, with=FALSE]),label=data.matrix(train$class))
# watchlist<-list(val=dval,train=dtrain)

param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .05,
              max_depth=3,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=1
)
set.seed(201510)
(tme <- Sys.time())
xgb3 <- xgb.cv(data = dtrain,
                  params = param,
                  nrounds = 6000,
                  folds=cvFolds,
                  maximize=FALSE,
                  prediction=TRUE,
                  print.every.n = 20,
                  early.stop.round=50)
Sys.time() - tme
save(xgb3, file="./stack_models/xgb3.rda")

samp <- read.csv('sample_submission.csv')
cnames <- paste("xgb3", names(samp)[2:ncol(samp)], sep="_")
colnames(xgb3$pred) <- cnames
write.csv(data.frame(VisitNumber=ts1Trans[filter==0,"VisitNumber",with=FALSE], xgb3$pred), "./stack_models/cvPreds_xgb3.csv", row.names=FALSE) 

## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201510)
(tme <- Sys.time())
xgb3full <- xgb.train(data = dtrain,
               params = param,
               nrounds = 4200,
               maximize=FALSE,
               print.every.n = 20)
Sys.time() - tme
save(xgb3full, file="./stack_models/xgb3full.rda")

preds <- predict(xgb3full, data.matrix(ts1Trans[filter==2, varnames, with=FALSE]))
preds <- data.frame(t(matrix(preds, nrow=38, ncol=length(preds)/38)))
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/testPreds_xgb3full.csv", row.names=FALSE)

