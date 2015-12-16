library(caret)
library(plyr)
library(caretEnsemble)
setwd("/home/branden/Documents/kaggle/facebook")
train <- read.csv("train.csv")
test <- read.csv("test.csv")
bids <- read.csv("bids.csv")
trainCount <- read.csv("train_ip_url_counts.csv")
testCount <- read.csv("test_ip_url_counts.csv")

train <- merge(train, trainCount, by="bidder_id", all.x=TRUE)
test <- merge(test, testCount, by="bidder_id", all.x=TRUE)

all <- rbind(train[, !(colnames(train) %in% "outcome")], test)

# find last appearance of each auction (should be the winning bid)
bids$timediff <- ave(bids$time, factor(bids$bidder_id), FUN=function(x) c(NA,diff(x)))
bids$aucBidNum <- ave(1:nrow(bids), factor(bids$auction), FUN=function(x) 1:length(x) )
bids$aucBidDiff <- ave(bids$aucBidNum, factor(bids$bidder_id), factor(bids$auction), FUN=function(x) c(NA,diff(x)))

#save bids CSV because bids$aucBidDiff took hours to calculate (may be easier to do in SQL)
write.csv(bids, "bids2.csv")

lastBid <- bids[!duplicated(bids$auction, fromLast=TRUE),"bid_id"]
bids$lastBid <- ifelse(bids$bid_id %in% lastBid, 1,0)

firstBid <- bids[!duplicated(bids$auction, fromLast=FALSE),"bid_id"]
bids$firstBid <- ifelse(bids$bid_id %in% firstBid, 1,0)

bids$trainBid <- ifelse(bids$bidder_id %in% train$bidder_id, 1,0)

facCounts <- ddply(bids, .(bidder_id), summarise, deviceCount = length(unique(device)), auctionCount = length(unique(auction)),
                   countryCount = length(unique(country)), winBids = sum(lastBid), firstBids = sum(firstBid), bids=length(bid_id), avgTimeDiff = mean(timediff, na.rm=TRUE),
                   sdTimeDiff = sd(timediff, na.rm=TRUE), noTimeDiff=sum(ifelse(timediff==0,1,0),na.rm=TRUE), doubleBid=sum(ifelse(aucBidDiff==1,1,0), na.rm=TRUE))

auctCounts <- ddply(bids, .(bidder_id, auction), summarise, win = sum(lastBid), bidCount = length(bid_id))
auctCounts$oneBidWin <- ifelse(auctCounts$win==1 & auctCounts$bidCount==1, 1, 0)
oneBidWins <- ddply(auctCounts, .(bidder_id), summarise, oneBidWins = sum(oneBidWin))

allMerge <- merge(all, facCounts, by="bidder_id",  all.x=TRUE)
allMerge <- merge(allMerge, oneBid, by="bidder_id", all.x=TRUE)

allMerge$winPerc <- allMerge$winBids / allMerge$auctionCount
allMerge$winPercBids <- allMerge$winBids / allMerge$bids
allMerge$bidsAuct <- allMerge$bids / allMerge$auctionCount
allMerge$firstBidPerc <- allMerge$firstBids / allMerge$auctionCount
allMerge$deviceAuct <- allMerge$deviceCount / allMerge$auctionCount
allMerge$oneBid <- ifelse(allMerge$bids==1,1,0)
allMerge$avgTimeDiff <- log(allMerge$avgTimeDiff+1)
allMerge$sdTimeDiff <- log(allMerge$sdTimeDiff+1)
allMerge$noTimeDiffPerc <- allMerge$noTimeDiff / allMerge$bids
allMerge$doubleBidPerc <- allMerge$doubleBid / allMerge$bids
allMerge[is.na(allMerge)] <- 0

# determine the rate at which bidders get a "one bid, win"


trBids <- subset(bids, bids$bidder_id %in% train$bidder_id)
trBids <- merge(trBids, train[,c("bidder_id","outcome")], by="bidder_id", all.x=TRUE)
trBids <- trBids[order(trBids$bid_id, decreasing=FALSE),]
mean(log(trBids[trBids$outcome=="No",]$time))
str(trBids$outcome)

mean(training[training$outcome=="No",]$doubleBidPerc, na.rm=TRUE)
mean(training[training$outcome=="Yes",]$doubleBidPerc, na.rm=TRUE)
mean(trMerge[trMerge$outcome=="No",]$sdTimeDiff, na.rm=TRUE)
mean(trMerge[trMerge$outcome=="Yes",]$sdTimeDiff, na.rm=TRUE)

mean(log(trBids[trBids$outcome=="Yes",]$avgTimeDiff))
mean(log(trBids[trBids$outcome=="No",]$bidsAuct))
mean(log(trBids[trBids$outcome=="Yes",]$bidsAuct))
var(log(trBids[trBids$outcome=="No",]$time))
var(log(trBids[trBids$outcome=="Yes",]$time))


var(trMerge[trMerge$outcome=="No",]$bidsAuct)
var(trMerge[trMerge$outcome=="Yes",]$bidsAuct)
mean(trMerge[trMerge$outcome=="No",]$countryCount)
mean(trMerge[trMerge$outcome=="Yes",]$countryCount)
sum(trMerge[trMerge$outcome=="No",]$deviceCount)
sum(trMerge[trMerge$outcome=="No",]$auctionCount)
sum(trMerge[trMerge$outcome=="Yes",]$deviceCount)
sum(trMerge[trMerge$outcome=="Yes",]$auctionCount)
## EXPLORATION
botID <- subset(train, outcome=="Yes", select=bidder_id)
nonBotID <- subset(train, outcome=="No", select=bidder_id)

botBids <- subset(bids, bidder_id %in% botID$bidder_id)
nonBotBids <- subset(bids, bidder_id %in% nonBotID$bidder_id)

# Add a running (and reverse, to see who bid last) bid count for each auction to see if bots bid 1st, 2nd, or last a lot
# Add bid count for each bidder_id

# Remove payment_account & address -- REDUNDANT
allMerge <- allMerge[,-c(1:3)]
training <- head(allMerge, nrow(train))
training$outcome <- as.factor(ifelse(train$outcome==1,"Yes","No"))
testing <- tail(allMerge, nrow(test))




##


library(plyr)


trMerge <- trMerge[complete.cases(trMerge),]
head(trMerge[order(trMerge$count, decreasing=TRUE),])

View(trMerge)
View(bids[bids$bidder_id=="91a3c57b13234af24875c56fb7e2b2f4rb56a",])


summary(glm(as.factor(outcome) ~ ., data=trMerge, family="binomial"))


## GBM
library(caret)
library(doParallel)
gbmCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
gbmGrid <- expand.grid(interaction.depth=c(11,13,15,17,21), n.trees=c(2000), shrinkage=c(.001))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
gbmTrain <- train(outcome ~ ., 
                  data=training,
                  method="gbm",
                  metric="ROC",
                  trControl=gbmCtrl,
                  tuneGrid=gbmGrid,
                  verbose=FALSE)
stopCluster(cl)
Sys.time() - tme
gbmTrain
gbm.perf(gbmTrain$finalModel)

gbmPred <- predict(gbmTrain, trMerge)
summary(gbmPred)

predTest <- predict(gbmTrain, newdata=testing, type="prob")$Yes

submission = data.frame(bidder_id = test$bidder_id, prediction = predTest)
write.csv(submission, "gbmSubmit.csv", row.names=FALSE)

## RF
rfCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
rfGrid <- expand.grid(mtry=c(1,3,5,7))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
rfTrain <- train(outcome ~ ., 
                  data=training,
                  method="rf",
                  metric="ROC",
                  ntree=3000,
                  nodesize=1,
                  importance=TRUE,
                  trControl=rfCtrl,
                  tuneGrid=rfGrid)
stopCluster(cl)
Sys.time() - tme
rfTrain
importance(rfTrain$finalModel)

gbmPred <- predict(rfTrain, trMerge)
summary(gbmPred)

rfTest <- predict(rfTrain, newdata=testing, type="prob")$Yes

submission = data.frame(bidder_id = test$bidder_id, prediction = rfTest)

write.csv(submission, "rfSubmit2.csv", row.names=FALSE)

## SVM
svmCtrl <- trainControl(method="cv",
                       number=10,
                       classProbs=TRUE,
                       allowParallel=TRUE,
                       summaryFunction=fiveStats)
sigma <- sigest(outcome ~ ., data=training, frac=.8)
svmGrid <- expand.grid(.sigma=c(.05,.1219,.2), .C=c(1,2,4,8,16,32))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
svmTrain <- train(outcome ~ ., 
                 data=training,
                 method="svmRadial",
                 metric="ROC",
                 trControl=svmCtrl,
                 tuneGrid=svmGrid
                 )
stopCluster(cl)
Sys.time() - tme
svmTrain


svmPred <- predict(svmTrain, trMerge)
summary(gbmPred)

predTest <- predict(gbmTrain, newdata=testMerge, type="prob")$Yes

submission = data.frame(bidder_id = test$bidder_id, prediction = predTest)

write.csv(submission, "gbmSubmit1.csv", row.names=FALSE)


## GLMNET
glmCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
glmnetGrid <- expand.grid(.alpha=c(.001,.0001), .lambda=2^seq(-7,2,1))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
glmnetTrain <- train(outcome ~  ., 
                  data=training,
                  method="glmnet",
                  metric="ROC",
                  trControl=glmCtrl,
                  tuneGrid=glmnetGrid
)
stopCluster(cl)
Sys.time() - tme
glmnetTrain

glmTest <- predict(glmTrain, newdata=testMerge, type="prob")$Yes

submission = data.frame(bidder_id = test$bidder_id, prediction = glmTest)

write.csv(submission, "glmSubmit1.csv", row.names=FALSE)


## Caret Ensemble
ensCtrl <- trainControl(method="cv",
                        number=10,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        classProbs=TRUE,
                        index=createResample(training$outcome, 25),
                        selectionFunction="best",
                        summaryFunction=fiveStats)
glmnetGrid <- expand.grid(alpha=c(.0001), lambda=.03125)
rfGrid <- expand.grid(mtry=c(3))
gbmGrid <- expand.grid(n.trees=c(2000), interaction.depth=c(9,11,13), shrinkage=c(.001))
svmGrid <- expand.grid(.sigma=c(.12),.C=c(1,2,4))

cl <- makeCluster(7)
registerDoParallel(cl)
tme <- Sys.time()
model_list <- caretList(
  outcome ~ .,
  data=training,
  trControl=ensCtrl,
  metric="ROC",
  tuneList=list(
    rf=caretModelSpec(method="rf", tuneGrid=rfGrid, nodesize=1, ntree=2000),
    gbm=caretModelSpec(method="gbm", tuneGrid=gbmGrid),
    svm=caretModelSpec(method="svmRadial", tuneGrid=svmGrid),
    glmnet=caretModelSpec(method="glm", family="binomial"))
  )
stopCluster(cl)
Sys.time() - tme

save(model_list, file="model_list.rda")

xyplot(resamples(model_list))
modelCor(resamples(model_list))
greedy_ensemble <- caretEnsemble(model_list)
summary(greedy_ensemble)

library('caTools')
model_preds <- lapply(model_list, predict, newdata=testMerge, type='prob')
model_preds <- lapply(model_preds, function(x) x[,'Yes'])
model_preds <- data.frame(model_preds)
ens_preds <- predict(greedy_ensemble, newdata=testMerge)
model_preds$ensemble <- ens_preds


# Ensemble Stack
cl <- makeCluster(7)
registerDoParallel(cl)
tme <- Sys.time()
gbm_stack <- caretStack(
  model_list, 
  method='gbm',
  verbose=FALSE,
  tuneGrid=expand.grid(n.trees=c(2000), interaction.depth=c(1,3,5,7,13,17,21), shrinkage=c(.001)),
  metric='ROC',
  trControl=trainControl(
    method='cv',
    number=10,
    savePredictions=TRUE,
    classProbs=TRUE,
    allowParallel=TRUE,
    summaryFunction=twoClassSummary
  )
)
stopCluster(cl)
Sys.time() - tme
gbm_stack

model_preds2 <- model_preds
stackPred <- predict(gbm_stack, newdata=testing, type='prob')$Yes
#CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
#colAUC(model_preds2, testing$Class)


## Creat Submission File
EnsSubmission = data.frame(bidder_id = test$bidder_id, prediction = stackPred)

write.csv(EnsSubmission, "SubmissionGBMStack-rf-gbm-glmnet-svm.csv", row.names=FALSE)
