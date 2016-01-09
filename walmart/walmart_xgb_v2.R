library(readr)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
setwd("/home/branden/Documents/kaggle/walmart")

t1 <- read.csv("train.csv")
s1 <- read.csv("test.csv")

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
  if ("class" %in% colnames(data)) 
      {x <-data %>% 
        group_by(VisitNumber, class) %>% 
        summarise(n=length(VisitNumber),
              # freqDept=names(sort(table(t1$DepartmentDescription),decreasing=TRUE)[1]),
              uniqDept=length(unique(DepartmentDescription)),
              uniqFine=length(unique(FinelineNumber)),
              uniqUpc=length(unique(Upc)),
              fineDeptRatio=uniqFine/uniqDept,
              upcDeptRatio=uniqUpc/uniqDept,
              upcFineRatio=uniqUpc/uniqFine,
              deptEntropy=entropy(DepartmentDescription),
              fineEntropy=entropy(FinelineNumber),
              upcEntropy=entropy(Upc),
              purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
              returns = -sum(ifelse(ScanCount<0,ScanCount,0)),
              netScans = purchases + returns,
              returnRatio = returns / netScans,
              scansDeptRatio=netScans/uniqDept,
              scansFineRatio=netScans/uniqFine,
              scansUpcRatio=netScans/uniqUpc)}
      else {x <-data %>% 
        group_by(VisitNumber) %>% 
        summarise(n=length(VisitNumber),
                  # freqDept=names(sort(table(t1$DepartmentDescription),decreasing=TRUE)[1]),
                  uniqDept=length(unique(DepartmentDescription)),
                  uniqFine=length(unique(FinelineNumber)),
                  uniqUpc=length(unique(Upc)),
                  fineDeptRatio=uniqFine/uniqDept,
                  upcDeptRatio=uniqUpc/uniqDept,
                  upcFineRatio=uniqUpc/uniqFine,
                  deptEntropy=entropy(DepartmentDescription),
                  fineEntropy=entropy(FinelineNumber),
                  upcEntropy=entropy(Upc),
                  purchases = sum(ifelse(ScanCount>0,ScanCount,0)),
                  returns = -sum(ifelse(ScanCount<0,ScanCount,0)),
                  netScans = purchases + returns,
                  returnRatio = returns / netScans,
                  scansDeptRatio=netScans/uniqDept,
                  scansFineRatio=netScans/uniqFine,
                  scansUpcRatio=netScans/uniqUpc)}
    xWeekday <- dcast(VisitNumber~Weekday, value.var="ScanCount",
                     fun.aggregate = sum, data=data)
    xDept <- dcast(VisitNumber~DepartmentDescription, value.var="ScanCount",
                 fun.aggregate = sum, data=data)
    xFine <- dcast(VisitNumber~FinelineNumber, value.var="ScanCount",
                   fun.aggregate = sum, data=data)
 
  xRes$VisitNumber <- NULL
  xAgg <- cbind(x, xRes)
  return(xAgg)
}



t1Trans <- data_transform(t1)
s1Trans <- data_transform(s1)

# t1agg <- cbind(t1sum, t1res)

set.seed(2016)
h <- sample(nrow(t1Trans), 3000)

dval<-xgb.DMatrix(data=data.matrix(model.matrix(class ~. , data=t1Trans[h,2:ncol(t1Trans)])[,-1]),label=t1Trans[h,"class"])
dtrain<-xgb.DMatrix(data=data.matrix(model.matrix(class ~. , data=t1Trans[-h,2:ncol(t1Trans)])[,-1]),label=t1Trans[-h,"class"])
watchlist<-list(val=dval,train=dtrain)
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .05,
              max_depth=10,
              min_child_weight=3,
              subsample=.8,
              colsample_bytree=.7
)
set.seed(201510)
xgb2 <- xgb.train(data = dtrain,
                  params = param,
                  nrounds = 1000,
                  maximize=FALSE,
                  print.every.n = 1,
                  watchlist=watchlist,
                  early.stop.round=30)

varnames <- colnames(data.matrix(model.matrix(class ~. , data=t1Trans[h,2:ncol(t1Trans)])[,-1]))
xgb3Imp <- xgb.importance(feature_names = varnames, model=xgb2)


preds <- predict(xgb2, data.matrix(s1Trans[,2:ncol(s1Trans)]))
predsMat <- data.frame(t(matrix(preds, nrow=38, ncol=length(preds)/38)))
samp <- read_csv('sample_submission.csv') 
cnames <- names(samp)[2:ncol(samp)]
names(predsMat) <- cnames
# colnames(predsMat) <- paste0("TripType_",tripClasses$TripType)
submission <- data.frame(VisitNumber=s1Trans$VisitNumber, predsMat)
write.csv(submission, "submit-xgb2-10-31-2015.csv", row.names=FALSE)
