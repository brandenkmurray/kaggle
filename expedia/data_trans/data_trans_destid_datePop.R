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

#######################
## Load data
#######################
t1 <- fread("./train.csv", select=c("is_booking","cnt","hotel_cluster","date_time","srch_ci","srch_co","srch_destination_id"))
s1 <- fread("./test.csv", select=c("date_time","srch_ci","srch_co","srch_destination_id"))
# dest <- fread("./destinations.csv")

t1$id <- -1
s1 <- s1[,c("is_booking"):=1][,c("cnt", "hotel_cluster"):=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),pred0=Mode(t1$hotel_cluster), dummy="A", filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)

rm(t1, s1)
dates <- c("srch_ci","srch_co")
ts1[,c("date_time"):=lapply(.SD,function(x) as.POSIXct(x, tz="GMT", format="%Y-%m-%d %H:%M:%S")), .SDcols=c("date_time")][,(dates):=lapply(.SD, function(x) as.Date(x, tz="GMT", format="%Y-%m-%d")), .SDcols=dates][,date:=as.Date(date_time, tz="GMT", format="%Y-%m-%d")]


# Convert check-in and check-out dates beyond 2017 as NA
# ts1$srch_ci[ts1$srch_ci_year >= 2018] <- NA
# ts1$srch_co[ts1$srch_co_year >= 2018] <- NA

# Calculate # of bookings in last X days
popTab <- ts1[,list(pop=sum(is_booking)), by=list(srch_destination_id,date)]
date.range = popTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(popTab,srch_destination_id,date)
popTab <- popTab[CJ(unique(srch_destination_id),all.dates)][, ":=" (pop_destid_date_1day=rollapplyr(c(NA, pop), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_3day=rollapplyr(c(NA, pop), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_7day=rollapplyr(c(NA, pop), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_15day=rollapplyr(c(NA, pop), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_30day=rollapplyr(c(NA, pop), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_60day=rollapplyr(c(NA, pop), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                    pop_destid_date_90day=rollapplyr(c(NA, pop), 90, sum, na.rm=TRUE, partial=TRUE)[-.N], 
                                                                    pop_destid_date_3day_priorYr=rollapplyr(c(rep(NA,366), pop), 3, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                    pop_destid_date_7day_priorYr=rollapplyr(c(rep(NA,366), pop), 7, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                    pop_destid_date_date_30day_priorYr=rollapplyr(c(rep(NA,366), pop), 30, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))]), 
                                                                by="srch_destination_id"][!is.na(pop)]

ts1 <- merge(ts1[,c("id2","srch_destination_id","date"),with=F], popTab[,!"pop",with=FALSE], by=c("srch_destination_id","date"), all.x = TRUE, sort=FALSE)

fwrite(ts1, "./data_trans/data_trans_destid_datePop.csv")
