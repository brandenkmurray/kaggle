library(timeDate)

setwd("/home/branden/Documents/kaggle/rossman")
t1 <- read.csv("train.csv")
s1 <- read.csv("test.csv")
store <- read.csv("store.csv")

t1 <- merge(t1, store)
s1 <- merge(s1, store)

t1Sales <- t1$Sales
t1$Sales <- NULL
#Customers only appears in the training set
#Maybe try predicting customers and then relate that to sales?
t1Customers <- t1$Customers
t1$Customers <- NULL

t1$filter <- 0
s1$filter <- 2
t1$Sales <- t1Sales
s1$Sales <- -1
s1ID <- s1$Id
s1$Id <- NULL

ts1 <- rbind(t1, s1)

ts1[is.na(ts1)] <- -999


# Removed closed and 0-sale observations from the training set


ts1$Date <- as.Date(ts1$Date, format="%Y-%m-%d")
holidayDates <- c("AshWednesday", "GoodFriday", "EasterSunday", "ChristTheKing", "Advent1st", "Advent1st", "Advent3rd", "Advent4th", "ChristmasEve", "ChristmasDay", "BoxingDay", "NewYearsDay","DEAscension", "DECorpusChristi", "DEGermanUnity", "DEChristmasEve", "DENewYearsEve")

holidays <- NULL
for (i in holidayDates) {
  tmp <- holiday(2013:2015, Holiday=i)
  tmp <- as.character(as.Date(tmp@Data, format=tmp@format))
  holidays <- c(holidays, tmp)
}
holidays <- as.Date(holidays)

ts1$Holiday <- ifelse(ts1$Date %in% holidays, 1, 0)

RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

param <- list(  objective           = "reg:linear", 
                #booster = "gblinear",
                eta                 = 0.2, # 0.06, #0.01,
                max_depth           = 8, #changed from default of 8
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.7 # 0.7
                
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 600, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
                    early.stop.round    = 30,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)