library(timeDate)
library(zoo)
library(xgboost)
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
ts1$dummy <- as.factor("A")
ts1$prob0 <- mean(ts1$Sales[ts1$filter==0])
ts1[is.na(ts1)] <- -9999


# Removed closed and 0-sale observations from the training set

# Separate date elements
ts1$Date <- as.Date(ts1$Date, format="%Y-%m-%d")
ts1$Date_mon <- strptime(ts1$Date, "%Y-%m-%d")$mon
ts1$Date_mday <- strptime(ts1$Date, "%Y-%m-%d")$mday
ts1$Date_yearmon <- as.numeric(as.yearmon(strptime(ts1$Date, "%Y-%m-%d")))
# Create a new column which indicates additional holidays
holidayDates <- c("AshWednesday", "GoodFriday", "EasterSunday", "ChristTheKing", "Advent1st", "Advent1st", "Advent3rd", "Advent4th", "ChristmasEve", "ChristmasDay", "BoxingDay", "NewYearsDay","DEAscension", "DECorpusChristi", "DEGermanUnity", "DEChristmasEve", "DENewYearsEve")

holidays <- NULL
for (i in holidayDates) {
  tmp <- holiday(2013:2015, Holiday=i)
  tmp <- as.character(as.Date(tmp@Data, format=tmp@format))
  holidays <- c(holidays, tmp)
}
holidays <- as.Date(holidays)
ts1$Holiday <- ifelse(ts1$Date %in% holidays, 1, 0)

ts1$CompetitionYearMonth <- as.Date(paste(ts1$CompetitionOpenSinceYear, ts1$CompetitionOpenSinceMonth, 1, sep="-"))
ts1$CompetitiontYearMonth_daysdiff <- as.numeric(difftime(time1 = ts1$Date, time2 = ts1$CompetitionYearMonth, units="days"))
ts1$CompetitiontYearMonth_daysdiff[is.na(ts1$CompetitiontYearMonth_daysdiff)] <- -9999

ts1$Promo2SinceWeek[ts1$Promo2SinceWeek==-9999]<-NA
ts1$Promo2SinceYear[ts1$Promo2SinceYear==-9999]<-NA
ts1$promoYearWeek <- as.Date(paste(ts1$Promo2SinceYear, ts1$Promo2SinceWeek, 1, sep="-"), format="%Y-%W-%w")
ts1$promoYearWeek_daysdiff <- as.numeric(difftime(time1 = ts1$Date, time2 = ts1$promoYearWeek, units="days"))
ts1$promoYearWeek_daysdiff[is.na(ts1$promoYearWeek_daysdiff)] <- -9999

ts1$Promo2SinceWeek[is.na(ts1$Promo2SinceWeek)] <- -9999
ts1$Promo2SinceYear[is.na(ts1$Promo2SinceYear)] <- -9999

train <- ts1[ts1$filter==0,]
test <- ts1[ts1$filter==2,]
train <- train[!names(train) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]
test <- test[!names(test) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]



# Create RMPSE metric for training
RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

varnames <- names(train[!names(train) %in% "Sales"])

param <- list(eta                 = 0.05, 
              max_depth           = 8,
              subsample           = 0.7,
              colsample_bytree    = 0.7
            
)

dtrain<-xgb.DMatrix(data=data.matrix(train[,varnames]),label=train$Sales)
set.seed(42069)
tme <- Sys.time()
xgb2 <- xgb.train(data        = dtrain,
                 # label       = train$Sales,
                objective           = "reg:linear", 
                nrounds             = 3000, 
                verbose             = 1,
                print.every.n       = 1,
                params              = param,
                maximize            = FALSE,
                feval               = RMPSE
)
(runTime <- Sys.time() - tme)

salesPred <- predict(xgb2, data.matrix(test[,varnames]))
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-04-2015-xgb2.csv", row.names=FALSE)
