library(readr)
library(data.table)
options(java.parameters="-Xmx125g")
library(extraTrees)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(gtools)
setwd("/home/branden/Documents/kaggle/walmart")

# t1 <- data.table(read.csv("train.csv"))
# s1 <- data.table(read.csv("test.csv"))
# 
# tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
# t1 <- merge(t1, tripClasses, by="TripType")
# t1 <- t1[order(t1$VisitNumber),]
# TripType <- t1$TripType
# t1$TripType <- NULL
# 
# s1$class <- -1
# t1$filter <- 0
# s1$filter <- 2
# 
# 
# # list of top UPCs
# topUPC <- names(sort(which(table(t1$Upc)>10), decreasing=TRUE))
# t1$Upc2 <- ifelse(t1$Upc %in% topUPC, t1$Upc, "Other")
# s1$Upc2 <- ifelse(s1$Upc %in% topUPC, s1$Upc, "Other")
# 
# x2 <- dcast.data.table(t1, Upc2 ~ class, fun.aggregate=length, value.var = "ScanCount")
# x3 <- x2[,2:ncol(x2),with=FALSE]/rowSums(x2[,2:ncol(x2), with=FALSE])
# x3 <- cbind(x2$Upc2,x3)
# 
# rowMax <- apply(x3[,2:ncol(x3), with=FALSE], 1, max)
# topUPC <- x3$V1[which(rowMax>.60)]
# 
# topUPC2 <- names(sort(table(t1$Upc), decreasing=TRUE))[1:500]
# upcList <- unique(c(topUPC, topUPC2))
# 
# 
# t1$Upc2 <- ifelse(t1$Upc %in% upcList, t1$Upc, "Other")
# s1$Upc2 <- ifelse(s1$Upc %in% upcList, s1$Upc, "Other")
# 
# ts1 <- rbind(t1, s1)
# ts1[is.na(ts1)] <- -99
# 
# ts1$Returns <- -ts1$ScanCount
# ts1$Returns[ts1$Returns < 0] <- 0
# ts1$Purchases <- ts1$ScanCount
# ts1$Purchases[ts1$Purchases < 0] <- 0
# 
# rm(t1)
# rm(s1)
# gc()
# 
# entropy <- function(x) {
#   tab <- table(as.character(x))
#   e <- sum(log(seq(1,sum(tab))))
#   for (i in tab){
#     e <- e - sum(log(seq(1,i)))
#   }
#   return(e)
# }
# 
# entropy2 <- function(x, count) {
#   tmp <- data.frame(x=x, count=count)
#   tmp <- tmp[tmp$count>0,]
#   if (nrow(tmp)==0)
#     {return(-1)}
#   else {
#     tab <- aggregate(count ~ x, tmp, sum)
#     e <- sum(log(seq(1,sum(tab$count))))
#     for (i in tab$count){
#       e <- e - sum(log(seq(1,i)))
#     }
#     return(e)
#   }
# }
# 
# 
# data_transform <- function(data){
#   data$ScanCount <- as.numeric(data$ScanCount)
#   data$VisitNumber <- as.factor(data$VisitNumber)
#   data$FinelineNumber <- as.factor(data$FinelineNumber)
#   x <- data[, list(n=length(DepartmentDescription),
#                    uniqDept=length(unique(DepartmentDescription)),
#                    uniqFine=length(unique(FinelineNumber)),
#                    uniqUpc=length(unique(Upc)),
#                    deptEntropy=entropy(DepartmentDescription),
#                    fineEntropy=entropy(FinelineNumber),
#                    upcEntropy=entropy(Upc),
#                    deptEntropy2=entropy2(DepartmentDescription, Purchases),
#                    fineEntropy2=entropy2(FinelineNumber, Purchases),
#                    upcEntropy2=entropy2(Upc, Purchases),
#                    purchases = sum(Purchases),
#                    returns = sum(Returns),
#                    purchDepts = length(unique(DepartmentDescription[Purchases>0])),
#                    returnDepts = length(unique(DepartmentDescription[Returns>0])),
#                    purchFine = length(unique(FinelineNumber[Purchases>0])),
#                    returnFine = length(unique(FinelineNumber[Returns>0])),
#                    purchUpc = length(unique(Upc[Purchases>0])),
#                    returnUpc = length(unique(Upc[Returns>0])),
#                    netScans=sum(Purchases + Returns),
#                    maxScans=max(ScanCount),
#                    minScans=min(ScanCount),
#                    meanScans=mean(ScanCount),
#                    medScans = median(ScanCount)
# #                    modeScans = names(sort(-table(ScanCount)))[1],
# #                    modeDept = names(sort(-table(DepartmentDescription)))[1],
# #                    modeFine = names(sort(-table(FinelineNumber)))[1],
# #                    modeUpc = names(sort(-table(Upc)))[1]
#                    ), by=list(VisitNumber,class,filter)]
#   x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
#                       upcDeptRatio=uniqUpc/uniqDept,
#                       upcFineRatio=uniqUpc/uniqFine,
#                       returnRatio = returns / netScans,
#                       deptFineEntRatio=ifelse(is.infinite(deptEntropy/fineEntropy),0,deptEntropy/fineEntropy),
#                       deptUpcEntRatio=ifelse(is.infinite(deptEntropy/upcEntropy),0,deptEntropy/upcEntropy),
#                       fineUpcEntRatio=ifelse(is.infinite(fineEntropy/upcEntropy),0,fineEntropy/upcEntropy),
#                       deptFineEntRatio2=ifelse(is.infinite(deptEntropy2/fineEntropy2),0,deptEntropy2/fineEntropy2),
#                       deptUpcEntRatio2=ifelse(is.infinite(deptEntropy2/upcEntropy2),0,deptEntropy2/upcEntropy2),
#                       fineUpcEntRatio2=ifelse(is.infinite(fineEntropy2/upcEntropy2),0,fineEntropy2/upcEntropy2),
#                       scansDeptRatio=netScans/uniqDept,
#                       scansFineRatio=netScans/uniqFine,
#                       scansUpcRatio=netScans/uniqUpc)]
#     
#   xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="Purchases",
#                    fun.aggregate = sum, data=data)
#   xWeekday <- data.table(xWeekday[,"VisitNumber",with=FALSE], prop.table(as.matrix(xWeekday[,2:ncol(xWeekday), with=FALSE]),margin=1))
#   xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="Purchases",
#                fun.aggregate = sum, data=data)
#   xDept <- data.table(xDept[,"VisitNumber",with=FALSE], prop.table(as.matrix(xDept[,2:ncol(xDept), with=FALSE]),margin=1))
#   xFine <- dcast.data.table(VisitNumber~FinelineNumber, value.var="Purchases",
#                  fun.aggregate = sum, data=data)
#   xFine <- data.table(xFine[,"VisitNumber",with=FALSE], prop.table(as.matrix(xFine[,2:ncol(xFine), with=FALSE]),margin=1))
#   xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="Purchases",
#                             fun.aggregate = sum, data=data)
#   # xUpc <- data.table(xUpc[,"VisitNumber",with=FALSE], prop.table(as.matrix(xUpc[,2:ncol(xUpc), with=FALSE]),margin=1))
#  
#   xAgg <- merge(x, xWeekday, by="VisitNumber")
#   xAgg <- merge(xAgg, xDept, by="VisitNumber")
#   xAgg <- merge(xAgg, xFine, by="VisitNumber")
#   xAgg <- merge(xAgg, xUpc, by="VisitNumber")
#   return(xAgg)
# }
# 
# ts1Trans <- data_transform(ts1)
# # Reorder the data set so that the train set is at the top and then order by VisitNumber
# ts1Trans <- ts1Trans[order(filter, VisitNumber),]
# # Some entropy values were 0. This created some NAs for the entropy ratios
# ts1Trans[is.na(ts1Trans)] <- 0
# 
# ts1Trans$TripType38_helper <- ts1Trans$`GROCERY DRY GOODS` + ts1Trans$DAIRY + ts1Trans$`COMM BREAD` + ts1Trans$`PRE PACKED DELI`
# ts1Trans$TripType39_helper <- ts1Trans$`PETS AND SUPPLIES` + ts1Trans$`PERSONAL CARE` + ts1Trans$`HOUSEHOLD CHEMICALS/SUPP` + ts1Trans$BEAUTY + ts1Trans$`PHARMACY OTC`
# ts1Trans$TripType7_helper <- ts1Trans$BAKERY + ts1Trans$`COMM BREAD` + ts1Trans$DAIRY + ts1Trans$`DSD GROCERY` + ts1Trans$`FROZEN FOODS` + ts1Trans$`GROCERY DRY GOODS` + ts1Trans$`MEAT - FRESH & FROZEN` + ts1Trans$`PRE PACKED DELI` + ts1Trans$PRODUCE +  + ts1Trans$`SERVICE DELI`
# ts1Trans$TripType8_helper <- ts1Trans$DAIRY + ts1Trans$`DSD GROCERY` + ts1Trans$`PERSONAL CARE` + ts1Trans$BEAUTY + ts1Trans$`GROCERY DRY GOODS` + ts1Trans$`IMPULSE MERCHANDISE` + ts1Trans$PRODUCE
# ts1Trans$TripType9_helper <- ts1Trans$AUTOMOTIVE + ts1Trans$CELEBRATION + ts1Trans$`MENS WEAR` + ts1Trans$`OFFICE SUPPLIES`
# ts1Trans$TripType35_helpter <- ts1Trans$`CANDY, TOBACCO, COOKIES` + ts1Trans$`DSD GROCERY` + ts1Trans$`IMPULSE MERCHANDISE`
# ts1Trans$TripType36_helper <- ts1Trans$BEAUTY + ts1Trans$`PERSONAL CARE` + ts1Trans$`PHARMACY OTC` + ts1Trans$`PETS AND SUPPLIES`

ts1Trans <- data.table(read_csv("./stack_models/ts1Trans3_abs.csv"))

# Create a vector the variable names to be used
varnames <- names(which(sapply(ts1Trans[filter==0, 4:ncol(ts1Trans), with=FALSE], function(x) uniqueN(x))>1))
# Create a separate train set -- This isn't necessary because I'm doing CV now instead of a single validation set 
# TSNE
# set.seed(42)
# ts1TSNE <- ts1Trans[, varnames, with=FALSE]
# ts1TSNE <- Rtsne(as.matrix(ts1Trans[, varnames, with=FALSE]), check_duplicates=FALSE, pca=FALSE, perplexity=30, theta=0.5, dims=2)

# REDOING THIS BECAUSE YOU DIDN'T DO TSNE ON THE WHOLE DATASET

# Was only necessary for easier filtering of the validation set
train <- ts1Trans[filter==0, c("class",varnames), with=FALSE]
train$class <- as.factor(make.names(train$class))
# train <- cbind(train, ts1TSNE$Y)

# IMPORTANT -- keep the same seed so that the folds used in creating models are the same.
set.seed(2016)
cvFolds <- createFolds(train$class, k=5, list=TRUE, returnTrain=TRUE)


# Logloss function
LogLoss <- function(actual, predicted, eps=1e-15) {
  predicted[predicted < eps] <- eps;
  predicted[predicted > 1 - eps] <- 1 - eps;
  -1/nrow(actual)*(sum(actual*log(predicted)))
}

et1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(train), ncol=38)) 
colnames(et1_stack_preds) <- unique(train$class)
et1_stack_preds <- et1_stack_preds[,mixedorder(colnames(et1_stack_preds))]
logLossTable <- data.frame(fold=seq(1:length(cvFolds)), LogLoss=rep(0, length(cvFolds)))
(tme <- Sys.time())
for (i in 1:length(cvFolds)){
  set.seed(201510)
  et1 <- extraTrees(x=data.matrix(train[cvFolds[[i]],varnames, with=FALSE]),
                  y=train$class[cvFolds[[i]]],
                  ntree=3000,
                  mtry=3,
                  nodesize=20,
                  numRandomCuts=5,
                  numThreads=16
  )
  # et1_pred <- predict(et1, newdata=data.matrix(train[-cvFolds[[1]],varnames, with=FALSE]), probability=TRUE)
   # LogLoss(model.matrix(~as.factor(class)-1, train[-cvFolds[[1]],"class", with=FALSE]), et1_pred)
#   
  # et1_stack_preds[-cvFolds[[i]],] <- predict(et1, newdata=data.matrix(train[-cvFolds[[i]],varnames, with=FALSE]), probability=TRUE)
  tmp <- as.data.frame(predict(et1, newdata=data.matrix(train[-cvFolds[[i]],varnames, with=FALSE]), probability=TRUE))
  tmp <- tmp[,mixedorder(colnames(tmp))]
  et1_stack_preds[-cvFolds[[i]],colnames(tmp)] <- tmp
  actTmp <- as.data.frame(model.matrix(~as.factor(class)-1, train[-cvFolds[[i]],"class", with=FALSE]))
  actTmp <- actTmp[,mixedorder(colnames(actTmp))]
  logLossTable[i,2] <- LogLoss(actTmp, et1_stack_preds[-cvFolds[[i]],])
}
Sys.time() - tme
logLossTable

cvPreds <- et1_stack_preds
samp <- read.csv('sample_submission.csv')
cnames <- paste("et1", names(samp)[2:ncol(samp)], sep="_")
colnames(cvPreds) <- cnames
cvPreds <- cbind(ts1Trans[filter==0,"VisitNumber",with=FALSE], cvPreds)
write.csv(cvPreds, "./stack_models/cvPreds_et1.csv", row.names=FALSE)

## Create a model using the full dataset -- make predictions on test set for use in future stacking
set.seed(201510)
(tme <- Sys.time())
et1full <- extraTrees(x=data.matrix(train[,varnames, with=FALSE]),
                      y=train$class,
                      ntree=3000,
                      mtry=3,
                      nodesize=20,
                      numRandomCuts=5,
                      numThreads=16
)
Sys.time() - tme
prepareForSave(et1full)
save(et1full, file="./stack_models/et1full.rda")

preds <- predict(et1full, data.matrix(ts1Trans[filter==2, varnames, with=FALSE]), probability=TRUE)
preds <- preds[,mixedorder(colnames(preds))]
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/testPreds_et1full.csv", row.names=FALSE)

