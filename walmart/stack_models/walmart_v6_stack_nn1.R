library(readr)
library(data.table)
library(caret)
library(reshape2)
library(dplyr)
library(deepnet)
library(h2o)
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

# list of top FinelineNumbers
topFine <- names(sort(which(table(t1$FinelineNumber)>10), decreasing=TRUE))
t1$FinelineNumber2 <- ifelse(t1$FinelineNumber %in% topFine, t1$FinelineNumber, "Other")
s1$FinelineNumber2 <- ifelse(s1$FinelineNumber %in% topFine, s1$FinelineNumber, "Other")

x2 <- dcast.data.table(t1, FinelineNumber2 ~ class, fun.aggregate=length, value.var = "ScanCount")
x3 <- x2[,2:ncol(x2),with=FALSE]/rowSums(x2[,2:ncol(x2), with=FALSE])
x3 <- cbind(x2$FinelineNumber2,x3)

rowMax <- apply(x3[,2:ncol(x3), with=FALSE], 1, max)
topFine <- x3$V1[which(rowMax>.60)]

topFine2 <- names(sort(table(t1$FinelineNumber), decreasing=TRUE))[1:5000]
fineList <- unique(c(topFine, topFine2))

t1$FinelineNumber2 <- ifelse(t1$FinelineNumber %in% fineList, t1$FinelineNumber, "Other")
s1$FinelineNumber2 <- ifelse(s1$FinelineNumber %in% fineList, s1$FinelineNumber, "Other")

# list of top UPCs
topUPC <- names(sort(which(table(t1$Upc)>20), decreasing=TRUE))
t1$Upc2 <- ifelse(t1$Upc %in% topUPC, t1$Upc, "Other")
s1$Upc2 <- ifelse(s1$Upc %in% topUPC, s1$Upc, "Other")

x2 <- dcast.data.table(t1, Upc2 ~ class, fun.aggregate=length, value.var = "ScanCount")
x3 <- x2[,2:ncol(x2),with=FALSE]/rowSums(x2[,2:ncol(x2), with=FALSE])
x3 <- cbind(x2$Upc2,x3)

rowMax <- apply(x3[,2:ncol(x3), with=FALSE], 1, max)
topUPC <- x3$V1[which(rowMax>.60)]

topUPC2 <- names(sort(table(t1$Upc), decreasing=TRUE))[1:1000]
upcList <- unique(c(topUPC, topUPC2))

t1$Upc2 <- ifelse(t1$Upc %in% upcList, t1$Upc, "Other")
s1$Upc2 <- ifelse(s1$Upc %in% upcList, s1$Upc, "Other")

ts1 <- rbind(t1, s1)
ts1[is.na(ts1)] <- -99

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
  x <- data[, list(n=length(DepartmentDescription),
                   uniqDept=length(unique(DepartmentDescription)),
                   uniqFine=length(unique(FinelineNumber)),
                   uniqUpc=length(unique(Upc)),
                   deptEntropy=entropy(DepartmentDescription),
                   fineEntropy=entropy(FinelineNumber),
                   upcEntropy=entropy(Upc),
                   deptEntropy2=entropy2(DepartmentDescription, Purchases),
                   fineEntropy2=entropy2(FinelineNumber, Purchases),
                   upcEntropy2=entropy2(Upc, Purchases),
                   purchases = sum(Purchases),
                   returns = sum(Returns),
                   purchDepts = length(unique(DepartmentDescription[Purchases>0])),
                   returnDepts = length(unique(DepartmentDescription[Returns>0])),
                   purchFine = length(unique(FinelineNumber[Purchases>0])),
                   returnFine = length(unique(FinelineNumber[Returns>0])),
                   purchUpc = length(unique(Upc[Purchases>0])),
                   returnUpc = length(unique(Upc[Returns>0])),
                   netScans=sum(Purchases + Returns),
                   maxScans=max(ScanCount),
                   minScans=min(ScanCount),
                   meanScans=mean(ScanCount),
                   medScans = median(ScanCount)
                   #                    modeScans = names(sort(-table(ScanCount)))[1],
                   #                    modeDept = names(sort(-table(DepartmentDescription)))[1],
                   #                    modeFine = names(sort(-table(FinelineNumber)))[1],
                   #                    modeUpc = names(sort(-table(Upc)))[1]
  ), by=list(VisitNumber,class,filter)]
  x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                upcDeptRatio=uniqUpc/uniqDept,
                upcFineRatio=uniqUpc/uniqFine,
                returnRatio = returns / netScans,
                deptFineEntRatio=ifelse(is.infinite(deptEntropy/fineEntropy),0,deptEntropy/fineEntropy),
                deptUpcEntRatio=ifelse(is.infinite(deptEntropy/upcEntropy),0,deptEntropy/upcEntropy),
                fineUpcEntRatio=ifelse(is.infinite(fineEntropy/upcEntropy),0,fineEntropy/upcEntropy),
                deptFineEntRatio2=ifelse(is.infinite(deptEntropy2/fineEntropy2),0,deptEntropy2/fineEntropy2),
                deptUpcEntRatio2=ifelse(is.infinite(deptEntropy2/upcEntropy2),0,deptEntropy2/upcEntropy2),
                fineUpcEntRatio2=ifelse(is.infinite(fineEntropy2/upcEntropy2),0,fineEntropy2/upcEntropy2),
                scansDeptRatio=netScans/uniqDept,
                scansFineRatio=netScans/uniqFine,
                scansUpcRatio=netScans/uniqUpc)]
  
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="Purchases",
                               fun.aggregate = sum, data=data)
  # xWeekday <- data.table(xWeekday[,"VisitNumber",with=FALSE], prop.table(as.matrix(xWeekday[,2:ncol(xWeekday), with=FALSE]),margin=1))
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  # xDept <- data.table(xDept[,"VisitNumber",with=FALSE], prop.table(as.matrix(xDept[,2:ncol(xDept), with=FALSE]),margin=1))
  xFine <- dcast.data.table(VisitNumber~FinelineNumber2, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  # xFine <- data.table(xFine[,"VisitNumber",with=FALSE], prop.table(as.matrix(xFine[,2:ncol(xFine), with=FALSE]),margin=1))
  xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="Purchases",
                           fun.aggregate = sum, data=data)
  # xUpc <- data.table(xUpc[,"VisitNumber",with=FALSE], prop.table(as.matrix(xUpc[,2:ncol(xUpc), with=FALSE]),margin=1))
  
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

#########################################################
### SAVE ts1Trans because it takes so long to create ####
# write.csv(ts1Trans, "ts1Trans2.csv", row.names=FALSE)
ts1Trans <- data.table(read.csv("ts1Trans.csv"))
#########################################################

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
set.seed(2016)
cvFolds2 <- createFolds(ts1Trans[filter==0,class], k=5, list=FALSE)

# Only do KNN with summary variables and Departments
nnPre <- preProcess(ts1Trans[filter==0,varnames, with=FALSE], thresh=.95, method=c("center","scale","pca"))
t1nn <- predict(nnPre, ts1Trans[filter==0,varnames, with=FALSE])
# s1knn <- data.frame(matrix(rep(0, ncol(t1knn)*nrow(s1Trans)), ncol=ncol(t1knn), nrow=nrow(s1Trans)))
# colnames(s1knn) <- names(t1knn)
# s1knn[,colnames(s1knn) %in% colnames(t1knn)] <- s1Trans[,colnames(s1Trans) %in% colnames(s1knn),with=FALSE]
s1nn <- predict(nnPre, ts1Trans[filter==2,varnames, with=FALSE])

t1nn$class <- ts1Trans[filter==0, "class",with=FALSE]
t1nn$class <- as.factor(t1nn$class)
# t1nn$folds <- cvFolds2
yDum <- model.matrix(~class-1, t1nn)

localH2O = h2o.init(nthreads = -1)
t1nn_h2o <- as.h2o(t1nn)


nn1 <- h2o.deeplearning(x=varnames, y="class",
                        t1nn_h2o,
                        activation = "RectifierWithDropout",
                        hidden = c(512, 256),
                        hidden_dropout_ratios = c(0.2,0.2),
                        input_dropout_ratio = 0.05,
                        epochs = 400,
                        l1 = c(0.0005),
                        l2 = c(0.0005), 
                        rho = 0.99, 
                        epsilon = 1e-8,
                        rate = 0.005,
                        rate_decay = 0.95,
                        quiet_mode=FALSE,
                        fold_column="folds",
                        keep_cross_validation_predictions = TRUE)



predList <- sapply(1:5, function(v) h2o.getFrame(nn1@model$cross_validation_predictions[[v]]$name), simplify = FALSE)
nn1_stack_preds <- as.data.frame(matrix(0, nrow=nrow(t1nn), ncol=38))
for (i in 1:5) {
  tmp <- as.data.frame(h2o.getFrame(nn1@model$cross_validation_predictions[[i]]$name)[-1])
  nn1_stack_preds <- nn1_stack_preds + tmp[,mixedorder(colnames(tmp))]
}
                    
nn1full <- h2o.deeplearning(x=varnames, y="class",
                        t1nn_h2o,
                        activation = "RectifierWithDropout",
                        hidden = c(512, 256),
                        hidden_dropout_ratios = c(0.2,0.2),
                        input_dropout_ratio = 0.05,
                        epochs = 400,
                        l1 = c(0.0005),
                        l2 = c(0.0005), 
                        rho = 0.99, 
                        epsilon = 1e-8,
                        rate = 0.005,
                        rate_decay = 0.95,
                        quiet_mode=FALSE)

tme <- Sys.time()
set.seed(1234)
nn1 <- nn.train(x = as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]),
         y = yDum[cvFolds[[5]],],
         hidden = c(128),
         activationfun = "sigm",
         learningrate = 0.05,
         learningrate_scale = .95,
         momentum = 0.9,
         numepochs = 256,
         batchsize = 200, 
         hidden_dropout=.2,
         output = "softmax")
Sys.time() - tme

nnTrainPreds <- nn.predict(nn1, as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[cvFolds[[5]],"class", with=FALSE]), nnTrainPreds)
# nnTest <- nn.test(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]), y=yDum[-cvFolds[[5]],])
nnPreds <- nn.predict(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[-cvFolds[[5]],"class", with=FALSE]), nnPreds)

tme <- Sys.time()
set.seed(1234)
nn2 <- nn.train(x = as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]),
                y = yDum[cvFolds[[5]],],
                hidden = c(64),
                activationfun = "sigm",
                learningrate = 0.05,
                learningrate_scale = .95,
                momentum = 0.5,
                numepochs = 64,
                batchsize = 100, 
                hidden_dropout=0,
                output = "softmax")
Sys.time() - tme

nnTrainPreds <- nn.predict(nn2, as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[cvFolds[[5]],"class", with=FALSE]), nnTrainPreds)
# nnTest <- nn.test(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]), y=yDum[-cvFolds[[5]],])
nnPreds <- nn.predict(nn2, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[-cvFolds[[5]],"class", with=FALSE]), nnPreds)




tme <- Sys.time()
dnn1 <- dbn.dnn.train(x = as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]), 
              y = yDum[cvFolds[[5]],], 
              hidden = c(64), 
              activationfun = "sigm", 
              learningrate = 0.05,
              momentum = 0.9, 
              learningrate_scale = .95, 
              output = "softmax", 
              # sae_output = "softmax",
              numepochs = 64, 
              batchsize = 100, 
              hidden_dropout = 0.3, 
              visible_dropout = 0)
Sys.time() - tme


nnTrainPreds <- nn.predict(nn1, as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[cvFolds[[5]],"class", with=FALSE]), nnTrainPreds)
# nnTest <- nn.test(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]), y=yDum[-cvFolds[[5]],])
nnPreds <- nn.predict(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[-cvFolds[[5]],"class", with=FALSE]), nnPreds)

library(nnet)
tme <- Sys.time()
set.seed(1234)
nnet1 <- nnet(x = as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]), 
              y = yDum[cvFolds[[5]],],
              size=48,
              softmax=T,
              decay=0,
              maxit=100,
              trace=T,
              MaxNWts=105000,
              abstol=1.0e-4,
              reltol = 1.0e-8
              )
Sys.time() - tme

nnTrainPreds <- predict(nnet1, as.matrix(t1nn[cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[cvFolds[[5]],"class", with=FALSE]), nnTrainPreds)
# nnTest <- nn.test(nn1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]), y=yDum[-cvFolds[[5]],])
nnPreds <- predict(nnet1, as.matrix(t1nn[-cvFolds[[5]],varnames, with=FALSE]))
LogLoss(model.matrix(~class-1, t1nn[-cvFolds[[5]],"class", with=FALSE]), nnPreds)

kknn2_stack_preds <- matrix(0, nrow=nrow(t1kknn), ncol=38)
logLossTable <- data.frame(fold=seq(1:length(cvFolds)), LogLoss=rep(0, length(cvFolds)))
for (i in 1:length(cvFolds)){
  kknn2 <- kknn(as.factor(class) ~ .,
                train=t1kknn[cvFolds[[i]],], 
                test=t1kknn[-cvFolds[[i]],], 
                k=400, 
                distance=1,
                kernel="triweight")
  kknn2_stack_preds[-cvFolds[[i]],] <- kknn2$prob
  logLossTable[i,2] <- LogLoss(model.matrix(~class-1, t1kknn[-cvFolds[[i]],"class", with=FALSE]), kknn2$prob)
}