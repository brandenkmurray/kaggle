# Run "helpCols.R" first
# The version does not utilize the similarity matrices.
# na.rm=TRUE was added to prevent NAs from being calculated in dcast.
library(readr)
library(reshape2)
library(data.table)
library(plyr)
library(zoo)
library(caret)
library(e1071)
library(Matrix)
library(proxy)
library(qlcMatrix)
library(cccd)
library(igraph)
library(quanteda)
setwd("/home/branden/Documents/kaggle/airbnb")
t1 <- data.table(read.csv("./train_users_2.csv"))
s1 <- data.table(read.csv("./test_users.csv"))
countries <- data.table(read.csv("./countries.csv"))
sess <- data.table(read.csv("./sessions.csv"))
age <- data.table(read.csv("./age_gender_bkts.csv"))
helpCols <- data.table(read.csv("./data_trans/helpCols.csv", stringsAsFactors = FALSE))

destClass <- data.frame(country_destination=sort(unique(t1$country_destination)), class=seq(0,11))
t1 <- merge(t1, destClass, by="country_destination")
t1 <- t1[order(t1$timestamp_first_active),]
country_destination <- t1$country_destination
t1$country_destination <- NULL

# write_csv(destClass, "./data_trans/classMap.csv")

s1$class <- -1
t1$filter <- 0
s1$filter <- 2

ts1 <- rbind(t1, s1)
class <- ts1$class
filter <- ts1$filter
ts1$class <- NULL
ts1$filter <- NULL
ts1 <- cbind(class, filter, ts1)
ts1$signup_flow <- as.factor(make.names(ts1$signup_flow))
ts1$date_first_booking <- NULL

ts1$createdDayOfWeek <- as.factor(weekdays(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdMonth <- as.factor(months(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdYear <- as.factor(year(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdDayOfMonth <- strptime(ts1$date_account_created, "%Y-%m-%d")$mday
ts1$createdYearMon <- as.numeric(as.yearmon(strptime(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdDayOfYear <- strptime(ts1$date_account_created, "%Y-%m-%d")$yday

ts1$activeYear <- as.factor(year(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeMonth <- as.factor(months(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeDayOfWeek <- as.factor(weekdays(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeDayOfMonth <- strptime(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")$mday
ts1$activeYearMon <- as.numeric(as.yearmon(strptime(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeDayOfYear <- strptime(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")$yday

ts1$daysDiffCreatedAct <- as.numeric(as.Date(ts1$date_account_created, "%Y-%m-%d") - as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S"))

ts1$date_account_created <- NULL
ts1$timestamp_first_active <- NULL

ts1[ts1$age < 14 |ts1$age > 100,'age'] <- -1
ts1[is.na(ts1)] <- -1

levels(ts1$first_affiliate_tracked)[levels(ts1$first_affiliate_tracked)==""] <- "NULL"

dummy <- dummyVars( ~. -1, data = ts1[,!colnames(ts1) %in% c("id"),with=FALSE])
ts1Dum <- data.frame(predict(dummy, ts1[,!colnames(ts1) %in% c("id"),with=FALSE]))
ts1Dum <- cbind(id=ts1$id, ts1Dum)

levels(sess$action)[levels(sess$action)==""] <- "NULL"
levels(sess$action_type)[levels(sess$action_type)==""] <- "NULL"
levels(sess$action_detail)[levels(sess$action_detail)==""] <- "NULL"
levels(sess$action)[levels(sess$action)=="-unknown-"] <- "unknown"
levels(sess$action_type)[levels(sess$action_type)=="-unknown-"] <- "unknown"
levels(sess$action_detail)[levels(sess$action_detail)=="-unknown-"] <- "unknown"
levels(sess$device_type)[levels(sess$device_type)=="-unknown-"] <- "unknown"

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
  {return(0)}
  else {
    tab <- aggregate(count ~ x, tmp, sum)
    e <- sum(log(seq(1,sum(tab$count))))
    for (i in tab$count){
      e <- e - sum(log(seq(1,i)))
    }
    return(e)
  }
}

simFunc <- function(data) {
  data <- data.table(data)
  cols <- ncol(data)
  dataSub <- Matrix(as.matrix(data[,2:cols, with=FALSE]))
  data_cosSparse <- as.matrix(cosSparse(dataSub))
  data_dist <- pr_simil2dist(data_cosSparse)
  data_dist_nng <- nng(dx=data_dist, k=4)
  
  V(data_dist_nng)$name <- rownames(data_cosSparse)
  E(data_dist_nng)$weight <- apply(get.edges(data_dist_nng,1:ecount(data_dist_nng)),1,function(x) data_cosSparse[x[1],x[2]])
  
  data_dist_adj <- as_adjacency_matrix(data_dist_nng, attr="weight")
  data_dist_adj_mat <- as.matrix(data_dist_adj)
  data_diag <- diag(x=1, nrow=nrow(data_dist_adj_mat))
  data_dist_adj_mat <- data_dist_adj_mat + data_diag
  
  data_dist_adj_mat <- data_dist_adj_mat %*% diag(1/rowSums(data_dist_adj_mat))
  
  data_simil <- as.matrix(dataSub %*% data_dist_adj_mat)
  data_simil_df <- data.frame(user_id=data[,"user_id",with=FALSE], data_simil)
  colnames(data_simil_df) <- colnames(data)
  return(data_simil_df)
}


sess_transform <- function(data){
  x <- data[, list(n=length(secs_elapsed),
                   uniqAction=length(unique(action)),
                   uniqActionType=length(unique(action_type)),
                   uniqActionDet=length(unique(action_detail)),
                   uniqDevice=length(unique(device_type)),
                   sumSecs = sum(secs_elapsed, na.rm=TRUE),
                   meanSecs = mean(secs_elapsed, na.rm=TRUE),
                   minSecs = ifelse(is.infinite(min(secs_elapsed, na.rm=TRUE)),-1,min(secs_elapsed, na.rm=TRUE)),
                   maxSecs = ifelse(is.infinite(max(secs_elapsed, na.rm=TRUE)),-1,max(secs_elapsed, na.rm=TRUE)),
                   medSecs = median(secs_elapsed, na.rm=TRUE),
                   sdSecs = ifelse(is.na(sd(secs_elapsed, na.rm=TRUE)),0,sd(secs_elapsed, na.rm=TRUE)),
                   skewSecs = ifelse(is.na(skewness(secs_elapsed, na.rm=TRUE)),0,skewness(secs_elapsed, na.rm=TRUE)),
                   kurtSecs = ifelse(is.na(kurtosis(secs_elapsed, na.rm=TRUE)),0,kurtosis(secs_elapsed, na.rm=TRUE)),
                   actionEnt = entropy(action),
                   actionTypeEnt = entropy(action_type),
                   actionDetEnt = entropy(action_detail),
                   deviceEnt = entropy(device_type),
                   firstSecs = secs_elapsed[1],
                   secondSecs = secs_elapsed[2],
                   thirdlastSecs = secs_elapsed[.N-2],
                   secondlastSecs = secs_elapsed[.N-1],
                   lastSecs = secs_elapsed[.N]
                   #                    actionEnt2 = entropy2(action, secs_elapsed),
                   #                    actionTypeEnt2 = entropy2(action_type, secs_elapsed),
                   #                    actionDetEng2 = entropy2(action_detail, secs_elapsed),
                   #                    deviceEnt2 = entropy2(device_type, secs_elapsed)
  ), by=list(user_id)]
  x <- x[, ':='(actTypeRatio=uniqActionType/uniqAction,
                actDetRatio=uniqActionDet/uniqAction,
                actDetTypeRatio=uniqActionDet/uniqActionType,
                actDevRatio=uniqAction/uniqDevice,
                actActTypeEntRatio=ifelse(is.infinite(actionEnt/actionTypeEnt),0,actionEnt/actionTypeEnt),
                actActDetEntRatio=ifelse(is.infinite(actionEnt/actionDetEnt),0,actionEnt/actionDetEnt),
                actDevEntRatio=ifelse(is.infinite(actionEnt/deviceEnt),0,actionEnt/deviceEnt),
                actTypeActDetEntRatio=ifelse(is.infinite(actionTypeEnt/actionDetEnt),0,actionTypeEnt/actionDetEnt),
                actTypeDevEntRatio=ifelse(is.infinite(actionTypeEnt/deviceEnt),0,actionTypeEnt/deviceEnt),
                actDetDevEntRatio=ifelse(is.infinite(actionDetEnt/deviceEnt),0,actionDetEnt/deviceEnt)
                #                 scansDeptRatio=netScans/uniqDept,
                #                 scansFineRatio=netScans/uniqFine,
                #                 scansUpcRatio=netScans/uniqUpc
  )]
  
  xAction <- dcast.data.table(user_id~action, value.var="secs_elapsed",
                              fun.aggregate = length, data=data)
  #   xAction <- data.table(xAction[,"user_id",with=FALSE], prop.table(as.matrix(xAction[,2:ncol(xAction), with=FALSE]),margin=1))
  #   xAction[is.na(xAction),] <- 0
  #   xActionSim <- simFunc(xAction)
  xActionSecs <- dcast.data.table(user_id~action, value.var="secs_elapsed",
                                  fun.aggregate = sum, na.rm=TRUE, data=data)
  #   xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  #   xActionSecs[is.na(xActionSecs),] <- 0
  #   xActionSecsSim <- simFunc(xActionSecs)
  #   xActionAvgSecs <- dcast.data.table(user_id~action, value.var="secs_elapsed",
  #                                   fun.aggregate = mean, na.rm=TRUE, data=data)
  # xActionAvgSecs <- data.table(xActionAvgSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionAvgSecs[,2:ncol(xActionAvgSecs), with=FALSE]),margin=1))
  
  
  xActionType <- dcast.data.table(user_id~action_type, value.var="secs_elapsed",
                                  fun.aggregate = length, data=data)
  #   xActionType <- data.table(xActionType [,"user_id",with=FALSE], prop.table(as.matrix(xActionType[,2:ncol(xActionType), with=FALSE]),margin=1))
  #   xActionType[is.na(xActionType),] <- 0
  #   xActionTypeSim <- simFunc(xActionType)
  xActionTypeSecs <- dcast.data.table(user_id~action_type, value.var="secs_elapsed",
                                      fun.aggregate = sum, na.rm=TRUE, data=data)
  #   xActionTypeSecs <- data.table(xActionTypeSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionTypeSecs[,2:ncol(xActionTypeSecs), with=FALSE]),margin=1))
  #   xActionTypeSecs[is.na(xActionTypeSecs),] <- 0
  #   xActionTypeSecsSim <- simFunc(xActionTypeSecs)
  #   xActionTypeAvgSecs <- dcast.data.table(user_id~action_type, value.var="secs_elapsed",
  #                                       fun.aggregate = mean, na.rm=TRUE, data=data)
  # xActionTypeAvgSecs <- data.table(xActionTypeAvgSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionTypeAvgSecs[,2:ncol(xActionTypeAvgSecs), with=FALSE]),margin=1))
  
  
  xActionDet <- dcast.data.table(user_id~action_detail, value.var="secs_elapsed",
                                 fun.aggregate = length, data=data)
  #   xActionDet <- data.table(xActionDet[,"user_id",with=FALSE], prop.table(as.matrix(xActionDet[,2:ncol(xActionDet), with=FALSE]),margin=1))
  #   xActionDet[is.na(xActionDet),] <- 0
  #   xActionDetSim <- simFunc(xActionDet)
  xActionDetSecs <- dcast.data.table(user_id~action_detail, value.var="secs_elapsed",
                                     fun.aggregate = sum, na.rm=TRUE, data=data)
  #   xActionDetSecs <- data.table(xActionDetSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionDetSecs[,2:ncol(xActionDetSecs), with=FALSE]),margin=1))
  #   xActionDetSecs[is.na(xActionDetSecs),] <- 0
  #   xActionDetSecsSim <- simFunc(xActionDetSecs)
  #   xActionDetAvgSecs <- dcast.data.table(user_id~action_detail, value.var="secs_elapsed",
  #                                      fun.aggregate = mean, na.rm=TRUE, data=data)
  # xActionDetAvgSecs <- data.table(xActionDetAvgSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionDetAvgSecs[,2:ncol(xActionDetAvgSecs), with=FALSE]),margin=1))  
  
  
  xDevice<- dcast.data.table(user_id~device_type, value.var="secs_elapsed",
                             fun.aggregate = length, data=data)
  #   xDevice <- data.table(xDevice[,"user_id",with=FALSE], prop.table(as.matrix(xDevice[,2:ncol(xDevice), with=FALSE]),margin=1))
  #   xDevice[is.na(xDevice),] <- 0
  #   xDeviceSim <- simFunc(xDevice)
  xDeviceSecs <- dcast.data.table(user_id~device_type, value.var="secs_elapsed",
                                  fun.aggregate = sum, na.rm=TRUE, data=data)
  #   xDeviceSecs <- data.table(xDeviceSecs[,"user_id",with=FALSE], prop.table(as.matrix(xDeviceSecs[,2:ncol(xDeviceSecs), with=FALSE]),margin=1))
  #   xDeviceSecs[is.na(xDeviceSecs),] <- 0
  #   xDeviceSecsSim <- simFunc(xDeviceSecs)
  #   xDeviceAvgSecs <- dcast.data.table(user_id~device_type, value.var="secs_elapsed",
  #                                   fun.aggregate = mean, na.rm=TRUE, data=data)
  # xDeviceAvgSecs <- data.table(xDeviceAvgSecs[,"user_id",with=FALSE], prop.table(as.matrix(xDeviceAvgSecs[,2:ncol(xDeviceAvgSecs), with=FALSE]),margin=1))
  
  xAgg <- merge(x, xAction, by="user_id", suffixes=c("summ","action"))
  xAgg <- merge(xAgg, xActionSecs, by="user_id", suffixes=c("","_actSecs"))
  # xAgg <- merge(xAgg, xActionAvgSecs, by="user_id", suffixes=c("","_actAvgSecs"))
  xAgg <- merge(xAgg, xActionType, by="user_id", suffixes=c("","_actType"))
  xAgg <- merge(xAgg, xActionTypeSecs, by="user_id", suffixes=c("","_actTypeSecs"))
  # xAgg <- merge(xAgg, xActionTypeAvgSecs, by="user_id", suffixes=c("","_actTypeAvgSecs"))
  xAgg <- merge(xAgg, xActionDet, by="user_id", suffixes=c("","_actDet"))
  xAgg <- merge(xAgg, xActionDetSecs, by="user_id", suffixes=c("","_actDetSecs"))
  # xAgg <- merge(xAgg, xActionDetAvgSecs, by="user_id", suffixes=c("","_actDetAvgSecs"))
  xAgg <- merge(xAgg, xDevice, by="user_id", suffixes=c("","_device"))
  xAgg <- merge(xAgg, xDeviceSecs, by="user_id", suffixes=c("","_deviceSecs"))  
  # xAgg <- merge(xAgg, xDeviceAvgSecs, by="user_id", suffixes=c("","_deviceAvgSecs")) 
  return(xAgg)
}

sess_ngrams <- function(data){
  x <- data[, list(actionString=toString(action),
                   actionTypeString=toString(action_type),
                   actionTypeDetString=toString(action_detail),
                   deviceString=toString(device_type)),
            by=list(user_id)]
  
  # Tokenize the strings
  actionTokens <- tokenize(toLower(x$actionString), removePunct = TRUE, ngrams = 2)
  actionTypeTokens <- tokenize(toLower(x$actionTypeString), removePunct = TRUE, ngrams = 2)
  actionTypeDetTokens <- tokenize(toLower(x$actionTypeDetString), removePunct = TRUE, ngrams = 2)
  deviceTokens <- tokenize(toLower(x$deviceString), removePunct = TRUE, ngrams = 2) 
  
  #Convert to DFM matrix
  actionDFM <- dfm(x = actionTokens)
  actionTypeDFM <- dfm(x = actionTypeTokens)
  actionTypeDetDFM <- dfm(x = actionTypeDetTokens)
  deviceDFM <- dfm(x = deviceTokens)
  
  # Find the top 100 most frequent Ngrams
  actionDFM_feats <- names(topfeatures(actionDFM, n=100))
  actionTypeDFM_feats <- names(topfeatures(actionTypeDFM, n=100))
  actionTypeDetDFM_feats <- names(topfeatures(actionTypeDetDFM, n=100))
  deviceDFM_feats <- names(topfeatures(deviceDFM, n=100))
  # Convert to a data.frame
  actionDF <- as.data.frame(actionDFM[,actionDFM_feats])
  colnames(actionDF) <- paste0(colnames(actionDF), "_actionNgram")
  actionTypeDF <- as.data.frame(actionTypeDFM[,actionTypeDFM_feats])
  colnames(actionTypeDF) <- paste0(colnames(actionTypeDF), "_actionTypeNgram")
  actionTypeDetDF <- as.data.frame(actionTypeDetDFM[,actionTypeDetDFM_feats])
  colnames(actionTypeDetDF) <- paste0(colnames(actionTypeDetDF), "_actionTypeDetNgram")
  deviceDF <- as.data.frame(deviceDFM[,deviceDFM_feats])
  colnames(deviceDF) <- paste0(colnames(deviceDF), "_deviceNgram")
  # Combine the data frames
  binded <- data.frame(user_id=x$user_id, actionDF, actionTypeDF, actionTypeDetDF, deviceDF)
  return(binded)
}


sessTrans <- sess_transform(sess)
# Replace NAs in the Seconds columns with -1
for (j in c("firstSecs", "secondSecs","thirdlastSecs","secondlastSecs","lastSecs"))
  set(sessTrans,which(is.na(sessTrans[[j]])),j,-1)
# sessNgram <- sess_ngrams(sess)
ts1_merge <- merge(ts1Dum, sessTrans, by.x="id",by.y="user_id", all.x=TRUE)
# ts1_merge <- merge(ts1_merge, sessNgram, by.x="id",by.y="user_id", all.x=TRUE)
ts1_merge[is.na(ts1_merge)] <- -1
ts1_merge <- ts1_merge[order(ts1_merge$id),]

# Convert NAs in Factor columns to "NA" factor level

# factors <- sapply(ts1_merge, is.factor)
# 
# ts1_merge[factors] <- lapply(ts1_merge[factors], function(x) {
#   levels(x) <- c(levels(x), 'NA')
#   x <- replace(x, which(is.na(x)), 'NA')
#   x <- droplevels(x)
# })

# dummies <- dummyVars(~.-1,data=ts1_merge[,!colnames(ts1_merge) %in% c("id","class","filter")])
# ts1_dum <- predict(dummies, newdata=ts1_merge)
# ts1_dum <- cbind(ts1_merge[,c("id","class","filter")], ts1_dum)

# pp <- preProcess(ts1_dum, method = c("medianImpute"))
# ts1_pp <- predict(pp, ts1_dum)
colnames(ts1_merge) <- sub(" ",".",names(ts1_merge))
colnames(ts1_merge) <- sub("-","",names(ts1_merge))
colnames(ts1_merge) <- gsub("`","",names(ts1_merge))

for (i in 0:11){
  if(length(grep("Secs",helpCols[[i+2]], value=TRUE, invert=TRUE)) > 1){
    ts1_merge[[ncol(ts1_merge)+1]] <- rowSums(ts1_merge[,grep("Secs",helpCols[[i+2]],value=TRUE, invert=TRUE)])
    colnames(ts1_merge)[ncol(ts1_merge)] <- paste0("X", i, "_helper")
  }
  if(length(grep("Secs",helpCols[[i+2]],value=TRUE)) > 1){
    ts1_merge[[ncol(ts1_merge)+1]] <- rowSums(ts1_merge[,grep("Secs",helpCols[[i+2]],value=TRUE)])
    colnames(ts1_merge)[ncol(ts1_merge)] <- paste0("X", i, "_helperSecs")
  }
}

pp <- preProcess(ts1_merge[ts1_merge$filter==0,4:ncol(ts1_merge)], method = c("zv","medianImpute", "center","scale"))
ts1_pp <- predict(pp, ts1_merge)


###############################################################
##### Golden features
###############################################################
gold_features <- function(df,number){
  list_out<-list()
  for(i in 1:number){
    list_out[[length(list_out)+1]]<-list(df[i,1],df[i,2])
  }
  list_out
}

actSecsCorDF <- abs(cor(ts1_pp[,grep("actSecs", colnames(ts1_pp))]))
actSecsCorDF[upper.tri(actSecsCorDF, diag=TRUE)] <- NA
actSecsCorDF <- melt(actSecsCorDF, varnames = c('V1','V2'), na.rm=TRUE)
actSecsCorDF <- actSecsCorDF[order(actSecsCorDF$value, decreasing=TRUE),]

actSecs_gold30 <- gold_features(actSecsCorDF, 30)

for (i in 1:length(actSecs_gold30)) {
  W=paste(actSecs_gold30[[i]][[1]],actSecs_gold30[[i]][[2]],sep="_")
  ts1_pp[,W] <- ts1_pp[,as.character(actSecs_gold30[[i]][[1]])] - ts1_pp[,as.character(actSecs_gold30[[i]][[2]])]
}


actTypeSecsCorDF <- abs(cor(ts1_pp[,grep("actTypeSecs", colnames(ts1_pp))]))
actTypeSecsCorDF[upper.tri(actTypeSecsCorDF, diag=TRUE)] <- NA
actTypeSecsCorDF <- melt(actTypeSecsCorDF, varnames = c('V1','V2'), na.rm=TRUE)
actTypeSecsCorDF <- actTypeSecsCorDF[order(actTypeSecsCorDF$value, decreasing=TRUE),]

actTypeSecs_gold5 <- gold_features(actTypeSecsCorDF, 5)

for (i in 1:length(actTypeSecs_gold5)) {
  W=paste(actTypeSecs_gold5[[i]][[1]],actTypeSecs_gold5[[i]][[2]],sep="_")
  ts1_pp[,W] <- ts1_pp[,as.character(actTypeSecs_gold5[[i]][[1]])] - ts1_pp[,as.character(actTypeSecs_gold5[[i]][[2]])]
}



actDetSecsCorDF <- abs(cor(ts1_pp[,grep("actDetSecs", colnames(ts1_pp))]))
actDetSecsCorDF[upper.tri(actDetSecsCorDF, diag=TRUE)] <- NA
actDetSecsCorDF <- melt(actDetSecsCorDF, varnames = c('V1','V2'), na.rm=TRUE)
actDetSecsCorDF <- actDetSecsCorDF[order(actDetSecsCorDF$value, decreasing=TRUE),]

actDetSecs_gold30 <- gold_features(actDetSecsCorDF, 30)

for (i in 1:length(actDetSecs_gold30)) {
  W=paste(actDetSecs_gold30[[i]][[1]],actDetSecs_gold30[[i]][[2]],sep="_")
  ts1_pp[,W] <- ts1_pp[,as.character(actDetSecs_gold30[[i]][[1]])] - ts1_pp[,as.character(actDetSecs_gold30[[i]][[2]])]
}

deviceSecsCorDF <- abs(cor(ts1_pp[,grep("deviceSecs", colnames(ts1_pp))]))
deviceSecsCorDF[upper.tri(deviceSecsCorDF, diag=TRUE)] <- NA
deviceSecsCorDF <- melt(deviceSecsCorDF, varnames = c('V1','V2'), na.rm=TRUE)
deviceSecsCorDF <- deviceSecsCorDF[order(deviceSecsCorDF$value, decreasing=TRUE),]

deviceSecs_gold30 <- gold_features(deviceSecsCorDF, 10)

for (i in 1:length(deviceSecs_gold30)) {
  W=paste(deviceSecs_gold30[[i]][[1]],deviceSecs_gold30[[i]][[2]],sep="_")
  ts1_pp[,W] <- ts1_pp[,as.character(deviceSecs_gold30[[i]][[1]])] - ts1_pp[,as.character(deviceSecs_gold30[[i]][[2]])]
}


write_csv(ts1_pp, "./data_trans/ts1_v12.csv")

