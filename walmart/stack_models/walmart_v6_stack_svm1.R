library(readr)
library(data.table)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(kernlab)
library(gtools)
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

ts1 <- rbind(t1, s1)
ts1[is.na(ts1)] <- -99

# list of top FinelineNumbers
topFine <- names(sort(which(table(ts1$FinelineNumber[ts1$filter==0])>50),decreasing=TRUE))
ts1$FinelineNumber2 <- ifelse(ts1$FinelineNumber %in% topFine, ts1$FinelineNumber, "Other")

x <- dcast.data.table(ts1[ts1$filter==0,], FinelineNumber2 ~ class, fun.aggregate=length, value.var="ScanCount")
x2 <- x[,2:ncol(x), with=FALSE]/rowSums(x[,2:ncol(x),with=FALSE])
x2 <- cbind(x$FinelineNumber2,x2)
rowMax <- apply(x2[,2:ncol(x2),with=FALSE], 1, max)
topFine2 <- x2$V1[which(rowMax>.4)]

topFine3 <- names(sort(table(ts1$FinelineNumber[ts1$filter==0]), decreasing=TRUE))[1:3000]
fineNames <- unique(c(topFine2, topFine3))

ts1$FinelineNumber2 <- ifelse(ts1$FinelineNumber %in% fineNames, ts1$FinelineNumber, "Other")

# list of top UPCs
topUpc <- names(sort(which(table(ts1$Upc[ts1$filter==0])>50),decreasing=TRUE))
ts1$Upc2 <- ifelse(ts1$Upc %in% topUpc, ts1$Upc, "Other")

x <- dcast.data.table(ts1[ts1$filter==0,], Upc2 ~ class, fun.aggregate=length, value.var="ScanCount")
x2 <- x[,2:ncol(x), with=FALSE]/rowSums(x[,2:ncol(x),with=FALSE])
x2 <- cbind(x$Upc2,x2)
rowMax <- apply(x2[,2:ncol(x2),with=FALSE], 1, max)
topUpc2 <- x2$V1[which(rowMax>.4)]

topUpc3 <- names(sort(table(ts1$Upc[ts1$filter==0]), decreasing=TRUE))[1:3000]
upcNames <- unique(c(topUpc2, topUpc3))

ts1$Upc2 <- ifelse(ts1$Upc %in% upcNames, ts1$Upc, "Other")

ts1$Returns <- -ts1$ScanCount


ts1$Returns[ts1$Returns < 0] <- 0
ts1$Purchases <- ts1$ScanCount 
ts1$Purchases[ts1$Purchases < 0] <- 0 

rm(t1)
rm(s1)
gc()

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
  x <- data[, list(n = length(DepartmentDescription),
                   uniqDept = length(unique(DepartmentDescription)),
                   uniqFine = length(unique(FinelineNumber)),
                   uniqUpc = length(unique(Upc)),
                   deptEntropy = entropy(DepartmentDescription),
                   fineEntropy = entropy(FinelineNumber),
                   upcEntropy = entropy(Upc),
                   deptEntropy2 = entropy2(DepartmentDescription, ScanCount),
                   fineEntropy2 = entropy2(FinelineNumber, ScanCount),
                   upcEntropy2 = entropy2(Upc, ScanCount),
                   purchases = sum(Purchases),
                   returns = sum(Returns),
                   netScans = sum(Purchases + Returns),
                   maxScans = max(ScanCount),
                   minScans = min(ScanCount),
                   meanScans = mean(ScanCount),
                   medScans = median(ScanCount),
                   modeScans = as.numeric(names(sort(-table(ScanCount)))[1])), by=list(VisitNumber,class,filter)]
  x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                      upcDeptRatio=uniqUpc/uniqDept,
                      upcFineRatio=uniqUpc/uniqFine,
                      returnRatio = returns / netScans,
                      deptFineEntRatio=ifelse(fineEntropy==0,0,deptEntropy/fineEntropy),
                      deptUpcEntRatio=ifelse(upcEntropy==0,0,deptEntropy/upcEntropy),
                      fineUpcEntRatio=ifelse(upcEntropy==0,0,fineEntropy/upcEntropy),
                      scansDeptRatio=netScans/uniqDept,
                      scansFineRatio=netScans/uniqFine,
                      scansUpcRatio=netScans/uniqUpc)]
    
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="Purchases",
                               fun.aggregate = sum, data=data)
  xWeekday <- data.table(xWeekday[,"VisitNumber",with=FALSE], prop.table(as.matrix(xWeekday[,2:ncol(xWeekday), with=FALSE]),margin=1))
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  xDept <- data.table(xDept[,"VisitNumber",with=FALSE], prop.table(as.matrix(xDept[,2:ncol(xDept), with=FALSE]),margin=1))
  xFine <- dcast.data.table(VisitNumber~FinelineNumber2, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  xFine <- data.table(xFine[,"VisitNumber",with=FALSE], prop.table(as.matrix(xFine[,2:ncol(xFine), with=FALSE]),margin=1))
  xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="Purchases",
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

rm(ts1)
gc()
# Create a vector the variable names to be used
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))
# Create a separate train set -- This isn't necessary because I'm doing CV now instead of a single validation set 
# Was only necessary for easier filtering of the validation set
# train <- ts1Trans[filter==0, c("class",varnames), with=FALSE]

# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}


# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
set.seed(2016)
cvFolds <- createFolds(ts1Trans[filter==0,class], k=5, list=TRUE, returnTrain=TRUE)

# Only do KNN with summary variables and Departments
svmPre <- preProcess(ts1Trans[filter==0,varnames, with=FALSE], pcaComp=3000, method=c("center","scale","pca"))
t1svm <- predict(svmPre, ts1Trans[filter==0,varnames, with=FALSE])
# s1knn <- data.frame(matrix(rep(0, ncol(t1knn)*nrow(s1Trans)), ncol=ncol(t1knn), nrow=nrow(s1Trans)))
# colnames(s1knn) <- names(t1knn)
# s1knn[,colnames(s1knn) %in% colnames(t1knn)] <- s1Trans[,colnames(s1Trans) %in% colnames(s1knn),with=FALSE]
s1svm <- predict(svmPre, ts1Trans[filter==2,varnames, with=FALSE])

t1svm$class <- ts1Trans[filter==0, "class",with=FALSE]
t1svm$class <- as.factor(make.names(t1svm$class))

y <- t1svm[cvFolds[[5]],"class", with=FALSE]$class


svm1 <- ksvm(class ~ ., 
             data=t1svm[cvFolds[[5]],],
#             x=t1svm[cvFolds[[5]],-c("class"), with=FALSE],
#              y=y,
             type="C-svc",
             kernel="rbfdot",
             prob.model = TRUE,
             kpar=list(sigma=0.00001),
             C=1)




library(e1071)
svm1 <- svm(class ~ ., 
             data=t1svm[cvFolds[[5]],],
             #             x=t1svm[cvFolds[[5]],-c("class"), with=FALSE],
             #              y=y,
             type="C-classification",
             kernel="radial",
             probability = TRUE,
             epsilon = 0.00001,
             C=128)

preds <- predict(svm1, newdata=t1svm[-cvFolds[[5]],], probability=TRUE)
predsFrame <- attr(preds, "probabilities")
predsFrame <- predsFrame[,mixedorder(colnames(predsFrame))]
acts <- model.matrix(~class-1, t1svm[-cvFolds[[5]],"class", with=FALSE])
acts <- acts[,mixedorder(colnames(acts))]
LogLoss(acts, predsFrame)



t1svm$class <- ts1Trans[filter==0, "class",with=FALSE]
t1svm$class <- as.factor(t1svm$class)


kknn3_stack_preds <- matrix(0, nrow=nrow(t1kknn), ncol=38)
logLossTable <- data.frame(fold=seq(1:length(cvFolds)), LogLoss=rep(0, length(cvFolds)))
for (i in 1:length(cvFolds)){
  kknn3 <- kknn(as.factor(class) ~ .,
                train=t1kknn[cvFolds[[i]],], 
                test=t1kknn[-cvFolds[[i]],], 
                k=400, 
                distance=1,
                kernel="triweight")
  kknn3_stack_preds[-cvFolds[[i]],] <- kknn3$prob
  logLossTable[i,2] <- LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[i]],"class", with=FALSE]), kknn3$prob)
}

cvPreds <- kknn3_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("kknn3", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"VisitNumber",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_kknn3.csv", row.names=FALSE)

# LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[5]],"class", with=FALSE]), kknn3_stack_preds$prob)
# LogLoss(model.matrix(~class-1, t1kknn[,"class", with=FALSE]), kknn2_stack_preds)


kknn3full <- kknn(as.factor(class) ~ .,
              train=t1kknn, 
              test=s1kknn, 
              k=400, 
              distance=1,
              kernel="triweight")
save(kknn3full, file="./stack_models/kknn3full.rda")

testPreds <- kknn3full$prob
colnames(testPreds) <- cnames
testPreds <- cbind(ts1Trans[filter==2,"VisitNumber",with=FALSE], testPreds)
write.csv(testPreds, "./stack_models/testPreds_kknn3full.csv", row.names=FALSE)
