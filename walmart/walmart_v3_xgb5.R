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

t1[is.na(t1)] <- -99
s1[is.na(s1)] <- -99
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
# t1$ScanCount <- as.numeric(t1$ScanCount)
# s1$ScanCount <- as.numeric(s1$ScanCount)
# t1$VisitNumber <- as.factor(t1$VisitNumber)
# s1$VisitNumber <- as.factor(s1$VisitNumber)

entropy <- function(x) {
  tab <- table(as.character(x))
  e <- sum(log(seq(1,sum(tab))))
  for (i in tab){
    e <- e - sum(log(seq(1,i)))
  }
  return(e)
}


data_transform <- function(data){
  data$ScanCount <- as.numeric(data$ScanCount)
  data$VisitNumber <- as.factor(data$VisitNumber)
  data$FinelineNumber <- as.factor(data$FinelineNumber)
  if ("class" %in% colnames(data)) 
      {x <- data[, list(n=length(DepartmentDescription),
                      uniqDept=length(unique(DepartmentDescription)),
                      uniqFine=length(unique(FinelineNumber)),
                      uniqUpc=length(unique(Upc)),
                      deptEntropy=entropy(DepartmentDescription),
                      fineEntropy=entropy(FinelineNumber),
                      upcEntropy=entropy(Upc),
                      purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
                      returns = -sum(ifelse(ScanCount<0,ScanCount,0)),
                      netScans=sum(abs(ScanCount))), by=list(VisitNumber,class)]
        x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                      upcDeptRatio=uniqUpc/uniqDept,
                      upcFineRatio=uniqUpc/uniqFine,
                      returnRatio = returns / netScans,
                      scansDeptRatio=netScans/uniqDept,
                      scansFineRatio=netScans/uniqFine,
                      scansUpcRatio=netScans/uniqUpc)]
      }
      else 
        {x <- data[, list(n=length(DepartmentDescription),
                                   uniqDept=length(unique(DepartmentDescription)),
                                   uniqFine=length(unique(FinelineNumber)),
                                   uniqUpc=length(unique(Upc)),
                                   deptEntropy=entropy(DepartmentDescription),
                                   fineEntropy=entropy(FinelineNumber),
                                   upcEntropy=entropy(Upc),
                                   purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
                                   returns = -sum(ifelse(ScanCount<0,ScanCount,0)),
                                   netScans=sum(abs(ScanCount))), by=list(VisitNumber)]
          x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                        upcDeptRatio=uniqUpc/uniqDept,
                        upcFineRatio=uniqUpc/uniqFine,
                        returnRatio = returns / netScans,
                        scansDeptRatio=netScans/uniqDept,
                        scansFineRatio=netScans/uniqFine,
                        scansUpcRatio=netScans/uniqUpc)]
          }
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="ScanCount",
                   fun.aggregate = sum, data=data)
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="ScanCount",
               fun.aggregate = sum, data=data)
  xFine <- dcast.data.table(VisitNumber~FinelineNumber, value.var="ScanCount",
                 fun.aggregate = sum, data=data)
 
  xWeekday[,VisitNumber:=NULL]
  xDept[,VisitNumber:=NULL]
  xFine[,VisitNumber:=NULL]
  xAgg <- cbind(x, xWeekday, xDept, xFine)
  # xAgg <- cbind(xWeekday, xDept, xFine)
  return(xAgg)
}

t1Trans <- data_transform(t1)
s1Trans <- data_transform(s1)
# t1class <- t1[, sum(length(class)), by=list(VisitNumber,class)][[2]]
# s1visit <- s1[, sum(length(VisitNumber)), by=list(VisitNumber)][[1]]                       
# t1agg <- cbind(t1sum, t1res)

# write_csv(t1Trans, "train_trans.csv")
# write_csv(s1Trans, "test_trans.csv")

set.seed(2017)
h <- sample(nrow(t1Trans), 2000)

# t1Trans$n <- as.numeric(t1Trans$n)
# t1Trans$uniqDept <- as.numeric(t1Trans$uniqDept)
# t1Trans$uniqFine <- as.numeric(t1Trans$uniqFine)
# t1Trans$uniqUpc <- as.numeric(t1Trans$uniqUpc)
# t1Trans$class <- as.numeric(t1Trans$class)

# varnames <- names(t1Trans[h,3:ncol(t1Trans), with=FALSE])
varnames <- read.csv("xgb4Imp.csv")$Feature

# Putting "colnames(t1Trans) %in% varnames" maintains the order of the columns -- makes it easier to deal with the test set
dval<-xgb.DMatrix(data=data.matrix(t1Trans[h,colnames(t1Trans) %in% varnames, with=FALSE]),label=data.matrix(t1Trans[h, "class", with=FALSE]))
dtrain<-xgb.DMatrix(data=data.matrix(t1Trans[-h,colnames(t1Trans) %in% varnames, with=FALSE]),label=data.matrix(t1Trans[-h, "class", with=FALSE]))
# dval<-xgb.DMatrix(data=data.matrix(t1Trans[h,1:ncol(t1Trans), with=FALSE]),label=t1class[h])
# dtrain<-xgb.DMatrix(data=data.matrix(t1Trans[-h,1:ncol(t1Trans), with=FALSE]),label=t1class[-h])

watchlist<-list(val=dval,train=dtrain)
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .05,
              max_depth=8,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=1
)
set.seed(201510)
(tme <- Sys.time())
xgb5 <- xgb.train(data = dtrain,
                  params = param,
                  nrounds = 3000,
                  maximize=FALSE,
                  print.every.n = 5,
                  watchlist=watchlist,
                  early.stop.round=50)
Sys.time() - tme
save(xgb5, file="xgb5.rda")

xgb5Imp <- xgb.importance(feature_names = varnames, model=xgb5)

write.csv(xgb5Imp, "xgb5Imp.csv")
# Create a new data.frame that contains all columns from the train set and excludes columns exclusive to the test set
# Make varnames vector same order as t1Trans order
varnamesOrd <- colnames(t1Trans[h,varnames, with=FALSE])
s1_new <- data.frame(matrix(rep(0, length(varnames)*nrow(s1Trans)), ncol=length(varnames), nrow=nrow(s1Trans)))
colnames(s1_new) <- names(t1Trans[h,colnames(t1Trans) %in% varnames, with=FALSE])
s1_new[,colnames(s1_new) %in% varnames] <- s1Trans[,colnames(s1Trans) %in% varnames,with=FALSE]
# s1_new[,colnames(s1_new) %in% colnames(s1Trans)] <- s1Trans[, varnames %in% colnames(s1Trans), with=FALSE]



preds <- predict(xgb5, data.matrix(s1_new))
preds <- data.frame(t(matrix(preds, nrow=38, ncol=length(preds)/38)))
samp <- read.csv('sample_submission.csv') 
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
# colnames(predsMat) <- paste0("TripType_",tripClasses$TripType)
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "submit-xgb5-11-1-2015.csv", row.names=FALSE)


