# Run "helpCols.R" first
library(readr)
library(data.table)
library(plyr)
library(zoo)
library(caret)
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

write_csv(destClass, "./data_trans/classMap.csv")

s1$class <- -1
t1$filter <- 0
s1$filter <- 2

ts1 <- rbind(t1, s1)
class <- ts1$class
filter <- ts1$filter
ts1$class <- NULL
ts1$filter <- NULL
ts1 <- cbind(class, filter, ts1)
ts1$date_first_booking <- NULL

ts1$createdDayOfWeek <- as.factor(weekdays(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdMonth <- as.factor(months(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdYear <- as.factor(year(as.Date(ts1$date_account_created, "%Y-%m-%d")))
ts1$createdDayOfMonth <- strptime(ts1$date_account_created, "%Y-%m-%d")$mday
ts1$createdYearMon <- as.numeric(as.yearmon(strptime(ts1$date_account_created, "%Y-%m-%d")))

ts1$activeYear <- as.factor(year(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeMonth <- as.factor(months(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeDayOfWeek <- as.factor(weekdays(as.Date(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))
ts1$activeDayOfMonth <- strptime(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")$mday
ts1$activeYearMon <- as.numeric(as.yearmon(strptime(as.character(ts1$timestamp_first_active), "%Y%m%d%H%M%S")))

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
                   actionEnt = entropy(action),
                   actionTypeEnt = entropy(action_type),
                   actionDetEnt = entropy(action_detail),
                   deviceEnt = entropy(device_type)
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
  # xAction <- data.table(xAction[,"user_id",with=FALSE], prop.table(as.matrix(xAction[,2:ncol(xAction), with=FALSE]),margin=1))
  xActionSecs <- dcast.data.table(user_id~action, value.var="secs_elapsed",
                              fun.aggregate = sum, data=data)
  # xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  xActionType <- dcast.data.table(user_id~action_type, value.var="secs_elapsed",
                              fun.aggregate = length, data=data)
  # xAction <- data.table(xAction[,"user_id",with=FALSE], prop.table(as.matrix(xAction[,2:ncol(xAction), with=FALSE]),margin=1))
  xActionTypeSecs <- dcast.data.table(user_id~action_type, value.var="secs_elapsed",
                                  fun.aggregate = sum, data=data)
  # xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  xActionDet <- dcast.data.table(user_id~action_detail, value.var="secs_elapsed",
                                  fun.aggregate = length, data=data)
  # xAction <- data.table(xAction[,"user_id",with=FALSE], prop.table(as.matrix(xAction[,2:ncol(xAction), with=FALSE]),margin=1))
  xActionDetSecs <- dcast.data.table(user_id~action_detail, value.var="secs_elapsed",
                                      fun.aggregate = sum, data=data)
  # xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  xDevice<- dcast.data.table(user_id~device_type, value.var="secs_elapsed",
                                     fun.aggregate = length, data=data)
  # xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  xDeviceSecs <- dcast.data.table(user_id~device_type, value.var="secs_elapsed",
                                     fun.aggregate = sum, data=data)
  # xActionSecs <- data.table(xActionSecs[,"user_id",with=FALSE], prop.table(as.matrix(xActionSecs[,2:ncol(xActionSecs), with=FALSE]),margin=1))
  
  xAgg <- merge(x, xAction, by="user_id", suffixes=c("summ","action"))
  xAgg <- merge(xAgg, xActionSecs, by="user_id", suffixes=c("","_actSecs"))
  xAgg <- merge(xAgg, xActionType, by="user_id", suffixes=c("","_actType"))
  xAgg <- merge(xAgg, xActionTypeSecs, by="user_id", suffixes=c("","_actTypeSecs"))
  xAgg <- merge(xAgg, xActionDet, by="user_id", suffixes=c("","_actDet"))
  xAgg <- merge(xAgg, xActionDetSecs, by="user_id", suffixes=c("","_actDetSecs"))
  xAgg <- merge(xAgg, xDevice, by="user_id", suffixes=c("","_device"))
  xAgg <- merge(xAgg, xDeviceSecs, by="user_id", suffixes=c("","_deviceSecs"))  
  return(xAgg)
}

sessTrans <- sess_transform(sess)
ts1_merge <- merge(ts1Dum, sessTrans, by.x="id",by.y="user_id", all.x=TRUE)
ts1_merge <- ts1_merge[order(ts1_merge$id),]

pp <- preProcess(ts1_merge, method = c("medianImpute"))
ts1_pp <- predict(pp, ts1_merge)
colnames(ts1_pp) <- sub(" ",".",names(ts1_pp))
colnames(ts1_pp) <- sub("-","",names(ts1_pp))

for (i in 0:11){
  if(length(grep("Secs",helpCols[[i+2]], value=TRUE, invert=TRUE)) > 1){
    ts1_pp[[ncol(ts1_pp)+1]] <- rowSums(ts1_pp[,grep("Secs",helpCols[[i+2]],value=TRUE, invert=TRUE)])
    colnames(ts1_pp)[ncol(ts1_pp)] <- paste0("X", i, "_helper")
  }
  if(length(grep("Secs",helpCols[[i+2]],value=TRUE)) > 1){
    ts1_pp[[ncol(ts1_pp)+1]] <- rowSums(ts1_pp[,grep("Secs",helpCols[[i+2]],value=TRUE)])
    colnames(ts1_pp)[ncol(ts1_pp)] <- paste0("X", i, "_helperSecs")
  }
}
write_csv(ts1_pp, "./data_trans/ts1_pp_v4.csv")

