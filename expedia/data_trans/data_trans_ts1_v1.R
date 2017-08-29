library(data.table)
library(zoo)
library(caret)
library(e1071)
library(Matrix)
# library(proxy)
# library(qlcMatrix)
# library(cccd)
# library(igraph)
library(gtools)
# library(plyr)
# library(dplyr)
# library(sqldf)
# library(DMwR)
# library(Rtsne)
library(doParallel)
library(doRNG)
# library(WGCNA)
# library(VGAM)
# library(Boruta)
library(readr)
library(RcppRoll)
library(Metrics)
library(stringr)
# setwd("/media/branden/SSHD1/kaggle/expedia")
# setwd("/media/branden/SSHD1/kaggle/expedia")
setwd("~/ebs")
# load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
source("./data_trans/utils.R")

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#ndcg metric from air's script
# map5 <- function(preds, dtrain) {
#   labels <- as.list(getinfo(dtrain,"label"))
#   num.class = 100
#   pred <- matrix(preds, nrow = num.class)
#   top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
#   top <- split(top, 1:NROW(top))
#   
#   map <- mapk(5, labels, top)
#   return(list(metric = "map5", value = map))
# }

map5 <- function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = t(matrix(preds, ncol = length(labels)))
  preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
  return (list(metric = 'map5', value = map5))
}
#######################
## Load data
#######################
t1 <- fread("./train.csv")
s1 <- fread("./test.csv")


t1$id <- -1
s1 <- s1[,c("is_booking"):=1][,c("cnt", "hotel_cluster"):=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),pred0=Mode(t1$hotel_cluster), dummy="A", filter=c(rep(0,31400000), rep(1,nrow(t1)-31400000), rep(2, nrow(s1))), ts1)


rm(t1, s1)
dates <- c("srch_ci","srch_co")
ts1[,c("date_time"):=lapply(.SD,function(x) as.POSIXct(x, tz="GMT", format="%Y-%m-%d %H:%M:%S")), .SDcols=c("date_time")][,(dates):=lapply(.SD, function(x) as.Date(x, tz="GMT", format="%Y-%m-%d")), .SDcols=dates][,date:=as.Date(date_time, tz="GMT", format="%Y-%m-%d")]

ts1 <- ts1[,':='(date_time_hour=hour(date_time),
                 date_time_wday=wday(date_time),
                 date_time_week=week(date_time),
                 date_time_month=month(date_time),
                 date_time_mday=mday(date_time),
                 date_time_yday=yday(date_time),
                 date_time_quarter=quarter(date_time),
                 date_time_year=year(date_time),
                 srch_ci_mday=mday(srch_ci),
                 srch_ci_yday=yday(srch_ci),
                 srch_ci_year=year(srch_ci),
                 srch_co_mday=mday(srch_co),
                 srch_co_yday=yday(srch_co)
)]



ts1[,":="(dist_clust_bookings_sum=sum(is_booking[filter %in% c(0,1)])-c(is_booking[filter %in% c(0,1)],rep(0, length(is_booking[filter %in% c(2)]))),
          dist_clust_cnts_sum=sum(cnt[filter %in% c(0,1)])-c(cnt[filter %in% c(0,1)],rep(0, length(cnt[filter %in% c(2)]))),
          dist_clust_bookings_count=length(is_booking[filter %in% c(0,1)])),
    keyby=list(orig_destination_distance)]
ts1[,":="(destid_clust_bookings_sum=sum(is_booking[filter %in% c(0,1)])-c(is_booking[filter %in% c(0,1)],rep(0, length(is_booking[filter %in% c(2)]))),
          destid_clust_cnts_sum=sum(cnt[filter %in% c(0,1)])-c(cnt[filter %in% c(0,1)],rep(0, length(cnt[filter %in% c(2)]))),
          destid_clust_bookings_count=length(is_booking[filter %in% c(0,1)])),
    keyby=list(srch_destination_id)]


# # Convert check-in and check-out dates beyond 2017 as NA
# ts1$srch_ci[ts1$srch_ci_year >= 2018] <- NA
# ts1$srch_co[ts1$srch_co_year >= 2018] <- NA
# # Calculate the cumulative number of times a hotel was booked
# # Calculate the number of times a hotel was booked in the previous X days, weeks, months, year
# 
# # ts2 <- copy(ts1)
# vars <- grep("bookings_srch_ci",colnames(ts1),value = TRUE)
# ts1[, (vars) := lapply(vars, function(x) {
#   x <- get(x)
#   x[is.na(x)] <- 0
#   x
# })]


# leak <- fread("./data_trans/leak_props.csv")
# 
# ts1 <- merge(ts1, leak, by=c("user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"), all.x=TRUE, sort=FALSE)
# rm(leak)
dates <- c("date_time", "srch_ci","srch_co","date")
ts1[,(dates):=NULL]
for (j in 2:ncol(ts1)){
  set(ts1, which(is.na(ts1[[j]])),j,-1)}

fwrite(ts1, "./data_trans/ts1_v1.csv")
