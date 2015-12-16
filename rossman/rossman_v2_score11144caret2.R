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

# train <- ts1[ts1$filter==0,]
# test <- ts1[ts1$filter==2,]
# train <- train[!names(train) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]
# test <- test[!names(test) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]

ts1 <- ts1[!names(ts1) %in% c("dummy","prob0","Date","CompetitionYearMonth","promoYearWeek")]
varnames <- c("Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval","Sales", "Date_mon", "Date_mday", "Date_yearmon", "Holiday", "CompetitiontYearMonth_daysdiff", "promoYearWeek_daysdiff")
trainInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==0, varnames]))[,-1]
testInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==2, varnames]))[,-1]
trainInd$Sales <- ts1$Sales[ts1$filter==0]
testInd$Sales <- ts1$Sales[ts1$filter==2]

# Remove closed stores and those without sales
# Closed and zero sale stores are exluded from scoring
# 0 sale stores in the train set mess up the RMPSE metric -- possible to fix?
trainInd <- trainInd[trainInd$Open==1,]
trainInd <- trainInd[trainInd$Sales!=0,]


# Create RMPSE metric for training
RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

varnames <- names(trainInd[!names(trainInd) %in% "Sales"])

param <- list(objective           = "reg:linear", 
              eta                 = 0.2, 
              max_depth           = 8,
              subsample           = 0.7,
              colsample_bytree    = 0.7
            
)

dtrain<-xgb.DMatrix(data=data.matrix(trainInd[,varnames]),label=log(trainInd$Sales+1))
set.seed(42069)
tme <- Sys.time()
xgb6 <- xgb.train(params              = param,
                  data        = dtrain,
                 # label       = train$Sales,
                 nrounds             = 600, 
                verbose             = 1,
                print.every.n       = 1,
                maximize            = FALSE,
                feval               = RMPSE
)
(runTime <- Sys.time() - tme)
save(xgb6, file="xgb6.rda")

salesPred <- exp(predict(xgb6, data.matrix(testInd[,varnames])))+1
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-04-2015-xgb6.csv", row.names=FALSE)


maeSummary <- function (data,
                        lev = NULL,
                        model = NULL) {
  out <- mae(data$obs, data$pred)  
  names(out) <- "MAE"
  out
}

RMPSEsummary <- function (data,
                          lev = NULL,
                          model = NULL) {
  elab<-exp(as.numeric(data[,"obs"]))-1
  epreds<-exp(as.numeric(data[,"pred"]))-1
  out <- sqrt(mean((epreds/elab-1)^2))
  names(out) <- "RMPSE"
  out
}

library(caret)
xgbCtrl <- trainControl(method="cv",
                        number=3,
                        summaryFunction = RMPSEsummary)
xgbGrid <- expand.grid(nrounds=c(2000), eta=c(.01), max_depth=c(20))
xgbGrid <- expand.grid(nrounds=c(10), eta=c(.2), max_depth=c(8))

tme <- Sys.time()
xgbCaret2 <- train(log(Sales+1) ~. ,
                   data=trainInd,
                   # x=trainInd[,varnames],
                   # y=log(trainInd$Sales+1),
                   method="xgbTree",
                   trControl=xgbCtrl,
                   tuneGrid = xgbGrid,
                   metric="RMPSE",
                   maximize=FALSE,
                   subsample           = 0.7,
                   colsample_bytree    = 0.7
                   )
(runTime <- Sys.time() - tme)
save(xgbCaret2, file="xgbCaret2.rda")
pbPost(type = "note", title = "xgbCaret2", body="Done.")

salesPred <- exp(predict(xgbCaret2, testInd))-1
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-05-2015-xgbCaret2.csv", row.names=FALSE)





grid <- expand.grid(eta=c(0.01), max_depth=c(8,12,16,20), nrounds=c(2000,5000,10000))
grid <- expand.grid(eta=c(0.01), max_depth=c(1,2,3,4), nrounds=c(10,20,30))
predFrame <- data.frame()
index <- 1
for (i in grid$eta){
  for (j in grid$max_depth){
    for (k in grid$nrounds){

      param <- list(objective           = "reg:linear", 
                    eta                 = i, 
                    max_depth           = j,
                    nrounds             = k,
                    subsample           = 0.7,
                    colsample_bytree    = 0.7
                    
      ) 
      
      xgbtmp <- xgb.train(params            = param,
                        data                = dtrain,
                        # label             = train$Sales,
                        verbose             = 1,
                        print.every.n       = 100,
                        maximize            = FALSE,
                        feval               = RMPSE
      )
      
      predFrame[,index] <- exp(predict(xgbtmp, data.matrix(testInd[,varnames])))+1
      index <- index + 1
    }
  }
}


