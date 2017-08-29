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
t1 <- fread("./train.csv", select=c("is_booking","cnt","hotel_cluster","date_time","srch_ci","srch_co","user_location_country","user_location_region","user_location_city","hotel_market"))
s1 <- fread("./test.csv", select=c("date_time","srch_ci","srch_co","user_location_country","user_location_region","user_location_city","hotel_market"))
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
# ts1[,leakCombos:=do.call("paste", c(ts1[,c("user_location_country","user_location_region","user_location_city","hotel_market"),with=FALSE], sep=""))]

# Convert check-in and check-out dates beyond 2017 as NA
# ts1$srch_ci[ts1$srch_ci_year >= 2018] <- NA
# ts1$srch_co[ts1$srch_co_year >= 2018] <- NA

# Calculate # of bookings in last X days
popTab <- ts1[,list(pop=sum(is_booking)), by=list(user_location_country,date)]
date.range = popTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(popTab,user_location_country,date)
popTabUserCountry <- popTab[CJ(unique(user_location_country),all.dates)][, ":=" (pop_userCountry_date_1day=rollapplyr(c(NA, pop), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_3day=rollapplyr(c(NA, pop), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_7day=rollapplyr(c(NA, pop), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_15day=rollapplyr(c(NA, pop), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_30day=rollapplyr(c(NA, pop), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_60day=rollapplyr(c(NA, pop), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userCountry_date_90day=rollapplyr(c(NA, pop), 90, sum, na.rm=TRUE, partial=TRUE)[-.N], 
                                                                      pop_userCountry_date_3day_priorYr=rollapplyr(c(rep(NA,366), pop), 3, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                      pop_userCountry_date_7day_priorYr=rollapplyr(c(rep(NA,366), pop), 7, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                      pop_userCountry_date_30day_priorYr=rollapplyr(c(rep(NA,366), pop), 30, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))]),
                                                              by="user_location_country"][!is.na(pop)]

popTab <- ts1[,list(pop=sum(is_booking)), by=list(user_location_region,date)]
date.range = popTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(popTab,user_location_region,date)
popTabUserRegion <- popTab[CJ(unique(user_location_region),all.dates)][, ":=" (pop_userRegion_date_1day=rollapplyr(c(NA, pop), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_3day=rollapplyr(c(NA, pop), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_7day=rollapplyr(c(NA, pop), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_15day=rollapplyr(c(NA, pop), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_30day=rollapplyr(c(NA, pop), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_60day=rollapplyr(c(NA, pop), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                      pop_userRegion_date_90day=rollapplyr(c(NA, pop), 90, sum, na.rm=TRUE, partial=TRUE)[-.N], 
                                                                      pop_userRegion_date_3day_priorYr=rollapplyr(c(rep(NA,366), pop), 3, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                      pop_userRegion_date_7day_priorYr=rollapplyr(c(rep(NA,366), pop), 7, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                      pop_userRegion_date_30day_priorYr=rollapplyr(c(rep(NA,366), pop), 30, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))]),
                                                              by="user_location_region"][!is.na(pop)]

popTab <- ts1[,list(pop=sum(is_booking)), by=list(user_location_city,date)]
date.range = popTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(popTab,user_location_city,date)
popTabUserCity <- popTab[CJ(unique(user_location_city),all.dates)][, ":=" (pop_userCity_date_1day=rollapplyr(c(NA, pop), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_3day=rollapplyr(c(NA, pop), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_7day=rollapplyr(c(NA, pop), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_15day=rollapplyr(c(NA, pop), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_30day=rollapplyr(c(NA, pop), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_60day=rollapplyr(c(NA, pop), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                           pop_userCity_date_90day=rollapplyr(c(NA, pop), 90, sum, na.rm=TRUE, partial=TRUE)[-.N], 
                                                                           pop_userCity_date_3day_priorYr=rollapplyr(c(rep(NA,366), pop), 3, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                           pop_userCity_date_7day_priorYr=rollapplyr(c(rep(NA,366), pop), 7, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                           pop_userCity_date_30day_priorYr=rollapplyr(c(rep(NA,366), pop), 30, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))]), 
                                                                   by="user_location_city"][!is.na(pop)]

popTab <- ts1[,list(pop=sum(is_booking)), by=list(hotel_market,date)]
date.range = popTab[,range(date)]
all.dates = seq.Date(date.range[1],date.range[2],by=1)
setkey(popTab,hotel_market,date)
popTabMarket <- popTab[CJ(unique(hotel_market),all.dates)][, ":=" (pop_Market_date_1day=rollapplyr(c(NA, pop), 1, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_3day=rollapplyr(c(NA, pop), 3, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_7day=rollapplyr(c(NA, pop), 7, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_15day=rollapplyr(c(NA, pop), 15, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_30day=rollapplyr(c(NA, pop), 30, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_60day=rollapplyr(c(NA, pop), 60, sum, na.rm=TRUE, partial=TRUE)[-.N],
                                                                   pop_Market_date_90day=rollapplyr(c(NA, pop), 90, sum, na.rm=TRUE, partial=TRUE)[-.N], 
                                                                   pop_Market_date_3day_priorYr=rollapplyr(c(rep(NA,366), pop), 3, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                   pop_Market_date_7day_priorYr=rollapplyr(c(rep(NA,366), pop), 7, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))],
                                                                   pop_Market_date_30day_priorYr=rollapplyr(c(rep(NA,366), pop), 30, sum, na.rm=TRUE, partial=TRUE)[-(.N:(.N-365))]),  
                                                             by="hotel_market"][!is.na(pop)]




ts1 <- merge(ts1[,c("id2","user_location_country","user_location_region","user_location_city","hotel_market","date"),with=F], popTabUserCountry[,!"pop",with=FALSE], by=c("user_location_country","date"), all.x = TRUE, sort=FALSE)
ts1 <- merge(ts1, popTabUserRegion[,!"pop",with=FALSE], by=c("user_location_region","date"), all.x = TRUE, sort=FALSE)
ts1 <- merge(ts1, popTabUserCity[,!"pop",with=FALSE], by=c("user_location_city","date"), all.x = TRUE, sort=FALSE)
ts1 <- merge(ts1, popTabMarket[,!"pop",with=FALSE], by=c("hotel_market","date"), all.x = TRUE, sort=FALSE)

fwrite(ts1, "./data_trans/data_trans_leak_datePop.csv")
