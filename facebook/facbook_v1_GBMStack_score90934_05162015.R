library(caret)
library(doParallel)
library(plyr)
library(reshape2)
library(caretEnsemble)
library(Hmisc)
library(flexclust)
setwd("/home/branden/Documents/kaggle/facebook")
train <- read.csv("train.csv")
test <- read.csv("test.csv")
bids <- read.csv("bids2.csv")
#bids <- read.csv("bids.csv")
trainCount <- read.csv("train_ip_url_counts.csv")
testCount <- read.csv("test_ip_url_counts.csv")
bidderAvgBids <- read.csv("bidders_avg_bids_auct_participated.csv")


train <- merge(train, trainCount, by="bidder_id", all.x=TRUE)
train <- merge(train, bidderAvgBids, by="bidder_id", all.x=TRUE)
test <- merge(test, testCount, by="bidder_id", all.x=TRUE)
test <- merge(test, bidderAvgBids, by="bidder_id", all.x=TRUE)

all <- rbind(train[, !(colnames(train) %in% "outcome")], test)

## Creat country variables for each bidder_id
levels(bids$country)[levels(bids$country)==""] <- "None"
bidsMelt <- melt(bids, id.vars="bidder_id", measure.vars="country")
bidsDcast <- dcast(bidsMelt, bidder_id ~ value, length)
bidsMelt1 <- melt(bidsDcast, id.vars="bidder_id")
bidsCountPerc <- ddply(bidsMelt1, .(bidder_id), transform, variable = variable, pct = value / sum(value))
bidsCount <- dcast(bidsCountPerc, bidder_id ~ variable, value.var="pct")

freqCountry <- aggregate(bids$country, list(bidder_id=bids$bidder_id), function(x) as.factor(names(table(x))[which.max(table(x))])) 
colnames(freqCountry)[2] <- "freqCountry"
freqCountry2 <- aggregate(bids$country, list(bidder_id=bids$bidder_id), function(x) as.factor(names(sort(table(x), decreasing=TRUE))[which.max(sort(table(x),decreasing=TRUE))+1]))
colnames(freqCountry2)[2] <- "freqCountry2"

topCountries <- names(sort(table(bids$country),decreasing=TRUE))[1:30]
bidsCountReduced <- bidsCount[,c("bidder_id", topCountries)]
bidsDcastReduced <- bidsDcast[,c("bidder_id", topCountries)]
bidsDcastBinary <- bidsDcastReduced[,2:ncol(bidsDcastReduced)]
bidsDcastBinary[bidsDcastBinary>0] <- 1
bidsDcastBinary$bidder_id <- bidsDcastReduced$bidder_id

# Get the first octet of the ip address
bids$ipSub <- as.factor(substr(bids$ip,1,regexpr(pattern="\\.", bids$ip)-1))
bidsIPMelt <- melt(bids, id.vars="bidder_id", measure.vars="ipSub")
bidsIPDcast <- dcast(bidsIPMelt, bidder_id ~ value, length)
freqIP <- aggregate(bids$ipSub, list(bidder_id=bids$bidder_id), function(x) as.factor(names(table(x))[which.max(table(x))])) 
colnames(freqIP)[2] <- "freqIP"



# find last appearance of each auction (should be the winning bid)
time <- bids$time
timeconst <- 52631578.95
time[1:2351187] <- time[1:2351187]/timeconst-time[1]/timeconst
time[2351188:5223204] <- time[2351188:5223204]/timeconst-time[2351188]/timeconst + time[2351187]+1
time[5223205:7656334] <- time[5223205:7656334]/timeconst-time[5223205]/timeconst + time[5223204]+1
bids$time <- time
bids$hours <- as.factor(floor(time/3600) %% 24)

# % of claims placed by hour
bidsHoursMelt <- melt(bids, id.vars="bidder_id", measure.vars="hours")
bidsHoursDcast <- dcast(bidsHoursMelt, bidder_id ~ value, length)
bidsHoursMelt1 <- melt(bidsHoursDcast, id.vars="bidder_id")
bidsHoursPerc <- ddply(bidsHoursMelt1, .(bidder_id), transform, variable = variable, pct = value / sum(value))
bidsHours <- dcast(bidsHoursPerc, bidder_id ~ variable, value.var="pct")
names(bidsHours)[-1] <- paste0("H",names(bidsHours)[-1])

bids$timediff <- ave(bids$time, factor(bids$bidder_id), FUN=function(x) c(NA,diff(x)))
bids$aucBidNum <- ave(1:nrow(bids), factor(bids$auction), FUN=function(x) 1:length(x) )
# # Takes too long to run, load "bids2.csv" instead to get this column if needed
# bids$aucBidDiff <- ave(bids$aucBidNum, factor(bids$bidder_id), factor(bids$auction), FUN=function(x) c(NA,diff(x)))
# #save bids CSV because bids$aucBidDiff took hours to calculate (may be easier to do in SQL)
# write.csv(bids, "bids2.csv", row.names=FALSE)

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
allMerge <- merge(allMerge, oneBidWins, by="bidder_id", all.x=TRUE)
#allMerge <- merge(allMerge, bidsDcastReduced, by="bidder_id", all.x=TRUE)
# allMerge <- merge(allMerge, freqIP, by="bidder_id", all.x=TRUE)
# allMerge <- merge(allMerge, freqCountry, by="bidder_id", all.x=TRUE)
# allMerge <- merge(allMerge, freqCountry2, by="bidder_id", all.x=TRUE)
allMerge <- merge(allMerge, bidsHours, by="bidder_id", all.x=TRUE)

allMerge$winPerc <- allMerge$winBids / allMerge$auctionCount
allMerge$winPercBids <- allMerge$winBids / allMerge$bids
allMerge$bidPercAuc <- allMerge$bids / allMerge$sumbids
allMerge$bidsAuct <- allMerge$bids / allMerge$auctionCount
allMerge$firstBidPerc <- allMerge$firstBids / allMerge$auctionCount
allMerge$deviceAuct <- allMerge$deviceCount / allMerge$auctionCount
allMerge$avgTimeDiff <- log(allMerge$avgTimeDiff+1)
# allMerge$avgTimeDiff <- cut2(allMerge$avgTimeDiff, g=10)
# levels(allMerge$avgTimeDiff) <- c(levels(allMerge$avgTimeDiff), "NA")
# allMerge$avgTimeDiff[is.na(allMerge$avgTimeDiff)] <- "NA"
allMerge$sdTimeDiff <- log(allMerge$sdTimeDiff+1)
# allMerge$sdTimeDiff <- cut2(allMerge$sdTimeDiff, g=10)
# levels(allMerge$sdTimeDiff) <- c(levels(allMerge$sdTimeDiff), "NA")
# allMerge$sdTimeDiff[is.na(allMerge$sdTimeDiff)] <- "NA"
allMerge$noTimeDiffPerc <- allMerge$noTimeDiff / allMerge$bids
allMerge$doubleBidPerc <- allMerge$doubleBid / allMerge$bids
allMerge[is.na(allMerge)] <- 0

# determine the rate at which bidders get a "one bid, win"


trBids <- subset(bids, bids$bidder_id %in% train$bidder_id)
trBids <- merge(trBids, train[,c("bidder_id","outcome")], by="bidder_id", all.x=TRUE)
trBids <- trBids[order(trBids$bid_id, decreasing=FALSE),]
mean(log(trBids[trBids$outcome=="No",]$time))
str(trBids$outcome)

trBidsSub <- subset(trBids, outcome==1)

bidsPlot <- ggplot(trBidsSub, aes(x=bidder_id, y=time))
bidsPlot + geom_jitter(aes(colour = bidder_id, alpha=.5)) + 
  theme(legend.position="none")


trBidsSub0 <- subset(trBids, outcome==0)
humans <- unique(trBids$bidder_id)
samphumans <- sample(humans, 100)
humanSub0 <- subset(trBids, outcome==0 & bidder_id %in% samphumans)
bidsPlot0 <- ggplot(humanSub0, aes(x=bidder_id, y=time))
bidsPlot0 + geom_jitter(aes(colour = bidder_id, alpha=.5)) + 
  theme(legend.position="none")

aucts <- unique(trBids$auction)
auctSamp <- sample(aucts, 200)
auctSub <- subset(trBids, auction %in% auctSamp)
auctTimePlot <- ggplot(auctSub, aes(x=auction, y=time))
auctTimePlot + geom_jitter(aes(colour = as.factor(outcome), alpha=.5)) + 
  theme(legend.position="none")


prop.table(table(trBids$hours, trBids$outcome),1)

View(table(trBids$ipSub, trBids$country))

ipBots <- ddply(trBids, .(ipSub, outcome=as.factor(outcome)), summarise, count=length(unique(bidder_id)))
ipBotsMut <- ddply(ipBots, .(ipSub), mutate, Perc = count/sum(count))
head(ipBots)

countryBots <- ddply(trBids, .(country, outcome=as.factor(outcome)), summarise, count=length(unique(bidder_id)))
countryBotsMut <- ddply(countryBots, .(country), mutate, Perc = count/sum(count))
head(ipBots)

hoursBots <- ddply(trBids, .(hours, outcome=as.factor(outcome)), summarise, count=length(unique(bidder_id)))
hoursBotsMut <- ddply(hoursBots, .(hours), mutate, Perc = count/sum(count))
head(hoursBots)

table(trBids$country, trBids$outcome)
prop.table(table(trBids$country, trBids$outcome), 1)

mean(bidsDcastTrain[bidsDcastTrain$outcome==0,]$de, na.rm=TRUE)
mean(bidsDcastTrain[bidsDcastTrain$outcome==1,]$de, na.rm=TRUE)

mean(training[training$outcome=="No",]$bidPercAuc, na.rm=TRUE)
mean(training[training$outcome=="Yes",]$bidPercAuc, na.rm=TRUE)
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
bidders <- unique(bids[bids$bidder_id %in% all$bidder_id, "bidder_id"])
training <- head(allMerge, nrow(train))
training$outcome <- as.factor(ifelse(train$outcome==1,"Yes","No"))
training <- training[training$bidder_id %in% bidders,-c(1:3)]
testing <- tail(allMerge, nrow(test))
testingID <- testing[testing$bidder_id %in% bidders,1]
testing <- testing[testing$bidder_id %in% bidders,-c(1:3)]




mean(training[training$outcome=="Yes" & training$bids > 1,"avgTimeDiff"])
mean(training[training$outcome=="No" & training$bids > 1,"avgTimeDiff"])





## K-means Cluster
trainClust <- training
trainClust$outcome <- NULL
clustSet <- rbind(trainClust, testing)
clustPP <- preProcess(clustSet, method=c("center","scale"))
normTrain <- predict(clustPP, clustSet)

distances <- dist(normTrain, method="euclidean")
clusterFB <- hclust(distances, method="ward.D")
plot(clusterFB)
clustGroups <- cutree(clusterFB, k=3)
table(clustGroups)

clustSetGroup <- cbind(clustSet, clustGroups)
clustTrain <- head(clustSetGroup,nrow(training))
clustTest <- tail(clustSetGroup,nrow(testing))
clustTest$bidder_id <- testingID

clustTrain$outcome <- training$outcome

clustTrain1 <- subset(clustTrain, clustGroups==1)
clustTrain2 <- subset(clustTrain, clustGroups==2)
clustTrain3 <- subset(clustTrain, clustGroups==3) 

clustTest1 <- subset(clustTest, clustGroups==1)
clustTest2 <- subset(clustTest, clustGroups==2)
clustTest3 <- subset(clustTest, clustGroups==3) 

ctr1NZV <- nearZeroVar(clustTrain1[,1:ncol(clustTrain)-1])
ctr2NZV <- nearZeroVar(clustTrain2[,1:ncol(clustTrain)-1])
ctr3NZV <- nearZeroVar(clustTrain3[,1:ncol(clustTrain)-1])

## GBM
library(caret)
library(doParallel)
gbmCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
gbmGrid <- expand.grid(interaction.depth=c(1,3,7,11,13,21), n.trees=c(2500), shrinkage=c(.001))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
set.seed(920)
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
gbm.perf(gbmTrain$finalModel, oobag.curve=TRUE)

gbmPred <- predict(gbmTrain, trMerge)
summary(gbmPred)

gbmTest <- predict(gbmTrain, newdata=testing, type="prob", na.action=na.pass)$Yes
gbmTable <- data.frame(bidder_id=testingID, gbmTest)
gbmTableMerge <- merge(tail(allMerge, nrow(test)), gbmTable, by="bidder_id", all.x=TRUE)

submission = data.frame(bidder_id = gbmTableMerge$bidder_id, prediction = gbmTableMerge$gbmTest)
submission[is.na(submission)] <- 0
write.csv(submission, "gbmSubmit3.csv", row.names=FALSE)

# Clustering Predict
gbmTrain1 <- gbmTrain
clustTest1$gbmPreds <- predict(gbmTrain1, newdata=clustTest1, type="prob")$Yes


gbmTrain2 <- gbmTrain
clustTest2$gbmPreds <- predict(gbmTrain2, newdata=clustTest2, type="prob")$Yes

gbmTrain3 <- gbmTrain
clustTest3$gbmPreds <- predict(gbmTrain3, newdata=clustTest3, type="prob")$Yes

gbmTestBind <- rbind(clustTest1, clustTest2, clustTest3)
gbmTable <- data.frame(bidder_id=gbmTestBind$bidder_id, gbmPreds=gbmTestBind$gbmPreds)
gbmTableMerge <- merge(tail(allMerge, nrow(test)), gbmTable, by="bidder_id", all.x=TRUE)
submission = data.frame(bidder_id = gbmTableMerge$bidder_id, prediction = gbmTableMerge$gbmPreds)
submission[is.na(submission)] <- 0
write.csv(submission, "gbmClusterSubmit.csv", row.names=FALSE)

## RF
rfCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
rfGrid <- expand.grid(mtry=c(3,5,7,11,15,17))

cl <- makeCluster(7)
registerDoParallel(cl)
set.seed(120)
tme <- Sys.time()
rfTrain <- train(outcome ~ ., 
                  data=training,
                  method="rf",
                  metric="ROC",
                  ntree=3000,
                  nodesize=1,
                  importance=TRUE,
                  trControl=rfCtrl,
                  tuneGrid=rfGrid
)
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
svmGrid <- expand.grid(.sigma=c(0.001, 0.01, 0.06, 0.2), .C=c(1,2,4,8,16,32))

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

## SVM Poly
svmCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
sigma <- sigest(outcome ~ ., data=training, frac=.8)
svmGrid <- expand.grid(.sigma=c(0.05, 0.1219, 0.2, 0.3, 0.4), .C=c(1,2,4,8,16,32))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
svmPolyTrain <- train(outcome ~ ., 
                  data=training,
                  method="svmPoly",
                  metric="ROC",
                  trControl=svmCtrl
)
stopCluster(cl)
Sys.time() - tme
svmPolyTrain


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

## NNET
nnetCtrl <- trainControl(method="cv",
                        number=10,
                        classProbs=TRUE,
                        allowParallel=TRUE,
                        summaryFunction=fiveStats)
nnetGrid <- expand.grid(.size=c(5,7,9,13), .decay=c(.001,.01,.1,.2))

cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
nnetTrain <- train(outcome ~  ., 
                     data=training,
                     method="nnet",
                     metric="ROC",
                     trControl=nnetCtrl,
                     tuneGrid=nnetGrid
)
stopCluster(cl)
Sys.time() - tme
nnetTrain

nnetTest <- predict(glmTrain, newdata=testMerge, type="prob")$Yes

submission = data.frame(bidder_id = test$bidder_id, prediction = nnetTest)

write.csv(submission, "glmSubmit1.csv", row.names=FALSE)

## Bayes GLM
cl <- makeCluster(6)
registerDoParallel(cl)
tme <- Sys.time()
bayGLMTrain <- train(outcome ~  ., 
                   data=training,
                   method="bayesglm",
                   metric="ROC",
                   trControl=nnetCtrl
)
stopCluster(cl)
Sys.time() - tme
bayGLMTrain

## Caret Ensemble
ensCtrl <- trainControl(method="cv",
                        number=10,
                        repeats=10,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        classProbs=TRUE,
                        index=createMultiFolds(training$outcome, k=10, times=10),
                        selectionFunction="best",
                        summaryFunction=fiveStats)
glmnetGrid <- expand.grid(alpha=c(.0001), lambda=.03125)
rfGrid <- expand.grid(mtry=c(3,5,7,11))
gbmGrid <- expand.grid(n.trees=c(2500), interaction.depth=c(5,7,9,11), shrinkage=c(.001))
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
    rf=caretModelSpec(method="rf", tuneGrid=rfGrid, nodesize=1, ntree=3000),
    gbm=caretModelSpec(method="gbm", tuneGrid=gbmGrid)
    #svm=caretModelSpec(method="svmRadial", tuneGrid=svmGrid),
    #glmnet=caretModelSpec(method="glm", family="binomial")
    )
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

# Not tuning stack
models <- caretList(rf=rfTrain, gbm=gbmTrain)
caretEns <- caretEnsemble(models)

# Ensemble Stack
cl <- makeCluster(7)
registerDoParallel(cl)
tme <- Sys.time()
gbm_stack <- caretStack(
  model_list, 
  method='gbm',
  verbose=FALSE,
  tuneGrid=expand.grid(n.trees=c(3000), interaction.depth=c(17,21,25,27,29,31), shrinkage=c(.001)),
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

gbm.perf(gbm_stack$ens_model$finalModel, oobag.curve=TRUE)

model_preds2 <- model_preds
stackPred <- predict(gbm_stack, newdata=testing, type='prob')$Yes
#CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
#colAUC(model_preds2, testing$Class)


gbmStackTest <- predict(gbm_stack, newdata=testing, type="prob", na.action=na.pass)$Yes
gbmTable <- data.frame(bidder_id=testingID, gbmStackTest)
gbmTableMerge <- merge(tail(allMerge, nrow(test)), gbmTable, by="bidder_id", all.x=TRUE)

submission = data.frame(bidder_id = gbmTableMerge$bidder_id, prediction = gbmTableMerge$gbmStackTest)
submission[is.na(submission)] <- 0

## Creat Submission File
write.csv(submission, "SubmissionGBMStack-rf-gbm05162015.csv", row.names=FALSE)



