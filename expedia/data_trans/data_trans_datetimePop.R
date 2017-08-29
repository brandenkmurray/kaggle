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
library(RcppRoll)
library(Metrics)
setwd("/media/branden/SSHD1/kaggle/expedia")
load("./data_trans/cvFoldsList.rda")
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
map5 <- function(preds, dtrain) {
  labels <- getinfo(dtrain,"label")
  num.class = 100
  pred <- matrix(preds, nrow = num.class)
  top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
  
  map <- mapk(5, labels, top)
  return(list(metric = "map5", value = map))
}

#######################
## Load data
#######################
t1 <- fread("./train.csv")
s1 <- fread("./test.csv")
dest <- fread("./destinations.csv")

t1$id <- -1
s1 <- s1[,c("is_booking"):=1][,c("cnt", "hotel_cluster"):=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),pred0=Mode(t1$hotel_cluster), dummy="A", filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)

rm(t1, s1)
dates <- c("srch_ci","srch_co")
ts1[,c("date_time"):=lapply(.SD,function(x) as.POSIXct(x, tz="GMT", format="%Y-%m-%d %H:%M:%S")), .SDcols=c("date_time")][,(dates):=lapply(.SD, function(x) as.Date(x, tz="GMT", format="%Y-%m-%d")), .SDcols=dates][,date:=as.Date(date_time, tz="GMT", format="%Y-%m-%d")]

ts1 <- ts1[,':='(date_time_hour=hour(date_time),
                 date_time_wday=wday(date_time),
                 date_time_week=week(date_time),
                 date_time_year=year(date_time),
                 date_time_month=month(date_time),
                 date_time_mday=mday(date_time),
                 date_time_yday=yday(date_time),
                 date_time_quarter=quarter(date_time),
                 srch_ci_wday=wday(srch_ci),
                 srch_ci_week=week(srch_ci),
                 srch_ci_year=year(srch_ci),
                 srch_ci_month=month(srch_ci),
                 srch_ci_mday=mday(srch_ci),
                 srch_ci_yday=yday(srch_ci),
                 srch_ci_quarter=quarter(srch_ci),
                 srch_co_wday=wday(srch_co),
                 srch_co_week=week(srch_co),
                 srch_co_year=year(srch_co),
                 srch_co_month=month(srch_co),
                 srch_co_mday=mday(srch_co),
                 srch_co_yday=yday(srch_co),
                 srch_co_quarter=quarter(srch_co)
)]
# Convert check-in and check-out dates beyond 2017 as NA
ts1$srch_ci[ts1$srch_ci_year >= 2018] <- NA
ts1$srch_co[ts1$srch_co_year >= 2018] <- NA
# Calculate the cumulative number of times a hotel was booked
# Calculate the number of times a hotel was booked in the previous X days, weeks, months, year
# data <- data.table(Date = rep(seq(as.Date("2014-05-01"),
#                                   as.Date("2014-09-01"),
#                                   by = 1),3),
#                    USD = rep(1:6, 7),
#                    ID = rep(c(1, 2), 21))
# data <- data[order(data$Date)]

# ts1Sub <- ts1[order(ts1$date_time)][1:500000]
# # setDT(ts1Sub)[,srch_destination_id2:=.GRP,by=c("srch_destination_id")]
# Ref <- ts1Sub[,list(Compare_Value=list(I(is_booking)),Compare_Date=list(I(date))), keyby=c("srch_destination_id2")]
# 
# # DT = as.data.table(ts1Sub) # this can be speedup by setDT()
# ts1Sub2 <- ts1Sub[,list(bookings=sum(is_booking)), by=list(srch_destination_id,date)]
# date.range = ts1Sub2[,range(date)]
# all.dates = seq.Date(date.range[1],date.range[2],by=1)
# setkey(ts1Sub2,srch_destination_id,date)
# r = ts1Sub2[CJ(unique(srch_destination_id),all.dates)][, c("roll") := rollapplyr(c(NA, bookings), 7, sum, na.rm=TRUE, partial=TRUE)[-.N], by="srch_destination_id"][!is.na(bookings)]
# 
# ts1Sub3 <- merge(ts1Sub, r, by=c("srch_destination_id", "date"), sort=FALSE)

bookingsTab <- ts1[,list(bookings=sum(is_booking)), by=list(srch_destination_id,date)]
date.range = bookingsTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(bookingsTab,srch_destination_id,date)
bookingsTab <- bookingsTab[CJ(unique(srch_destination_id),all.dates)][, ":=" (bookings_date_1day=rollapplyr(c(NA, bookings), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_3day=rollapplyr(c(NA, bookings), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_7day=rollapplyr(c(NA, bookings), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_15day=rollapplyr(c(NA, bookings), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_30day=rollapplyr(c(NA, bookings), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_60day=rollapplyr(c(NA, bookings), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                              bookings_date_90day=rollapplyr(c(NA, bookings), 90, sum, na.rm=TRUE, partial=TRUE)[-.N]), 
                                                                      by="srch_destination_id"][!is.na(bookings)]

ts1 <- merge(ts1, bookingsTab[,!"bookings",with=FALSE], by=c("srch_destination_id","date"), all.x = TRUE, sort=FALSE)

bookings_srch_ci_Tab <- ts1[,list(bookings=sum(is_booking)), by=list(srch_destination_id,srch_ci)]
date.range = bookings_srch_ci_Tab[,range(srch_ci, na.rm=TRUE)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(bookings_srch_ci_Tab,srch_destination_id,srch_ci)
bookings_srch_ci_Tab <- bookings_srch_ci_Tab[CJ(unique(srch_destination_id),all.dates)][, ":=" (bookings_srch_ci_1day=rollapplyr(c(NA, bookings), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_3day=rollapplyr(c(NA, bookings), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_7day=rollapplyr(c(NA, bookings), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_15day=rollapplyr(c(NA, bookings), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_30day=rollapplyr(c(NA, bookings), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_60day=rollapplyr(c(NA, bookings), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                                                bookings_srch_ci_90day=rollapplyr(c(NA, bookings), 90, sum, na.rm=TRUE, partial=TRUE)[-.N]), 
                                                                                        by="srch_destination_id"][!is.na(bookings)]

ts1 <- merge(ts1, bookings_srch_ci_Tab[,!"bookings",with=FALSE], by=c("srch_destination_id","srch_ci"), all.x=TRUE, sort=FALSE)
