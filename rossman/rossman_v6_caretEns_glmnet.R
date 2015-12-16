library(timeDate)
library(zoo)
library(xgboost)
library(stringr)
library(qdapTools)
library(sqldf)
setwd("/home/branden/Documents/kaggle/rossman")
source("/home/branden/Documents/kaggle/rossman/final__utils.R")
t1 <- read.csv("train.csv")
s1 <- read.csv("test.csv")
store <- read.csv("store.csv")

t1 <- merge(t1, store)
s1 <- merge(s1, store)
t1 <- t1[order(t1$Date, -t1$Store, decreasing=TRUE),]
s1 <- s1[order(s1$Date, -s1$Store, decreasing=TRUE),]

# t1Sales <- t1$Sales
# t1$Sales <- NULL
#Customers only appears in the training set
#Maybe try predicting customers and then relate that to sales?
t1Customers <- t1$Customers
t1$Customers <- NULL

t1$filter <- 0
s1$filter <- 2
# t1$Sales <- t1Sales
s1$Sales <- -1
s1ID <- s1$Id
s1$Id <- NULL

ts1 <- rbind(t1, s1)


ts1$dummy <- as.factor("A")
ts1$prob0 <- mean(ts1$Sales[ts1$filter==0])
ts1$Open[is.na(ts1$Open)] <- 0
ts1$CompetitionDistance[is.na(ts1$CompetitionDistance)] <- median(ts1$CompetitionDistance, na.rm=TRUE)
ts1$CompetitionYearMonth <- as.Date(paste(ts1$CompetitionOpenSinceYear, ts1$CompetitionOpenSinceMonth, 1, sep="-"))
ts1$CompetitionYearMonth_daysdiff <- as.numeric(difftime(time1 = as.Date("2015-10-01", format="%Y-%m-%d"), time2 = ts1$CompetitionYearMonth, units="days"))
ts1$CompetitionYearMonth_daysdiff[is.na(ts1$CompetitionYearMonth_daysdiff)] <- median(ts1$CompetitionYearMonth_daysdiff, na.rm=TRUE)
ts1$Promo2YearWeek <- as.Date(paste(ts1$Promo2SinceYear, ts1$Promo2SinceWeek, 1, sep="-"), format="%Y-%w-%d")
ts1$Promo2YearWeek_daysdiff <- as.numeric(difftime(time1 = as.Date("2015-10-01", format="%Y-%m-%d"), time2 = ts1$Promo2YearWeek, units="weeks"))
ts1$Promo2YearWeek_daysdiff[is.na(ts1$Promo2YearWeek_daysdiff)] <- median(ts1$Promo2YearWeek_daysdiff, na.rm=TRUE)


# varnames <- c("StateHoliday","StoreType","Assortment","PromoInterval")
# for (f in varnames) {
#   levels <- unique(ts1[[f]])
#   ts1[[f]] <- as.integer(factor(ts1[[f]], levels=levels))
#   
# }


# Separate date elements
ts1$Date <- as.Date(ts1$Date, format="%Y-%m-%d")
ts1$Date_numeric <- as.numeric(ts1$Date)
ts1$Date_year <- as.factor(strftime(ts1$Date, format="%Y"))
ts1$Date_mon <- as.factor(strptime(ts1$Date, "%Y-%m-%d")$mon)
ts1$Date_mday <- as.factor(strptime(ts1$Date, "%Y-%m-%d")$mday)
# ts1$Date_yearmon <- as.numeric(as.yearmon(strptime(ts1$Date, "%Y-%m-%d")))
ts1$Date_week <- as.factor(strftime(ts1$Date, format="%W"))
# Even/odd weeks -- if people are paid biweekly then this could affect purchase patterns
ts1$Date_evenweek <- ifelse(as.numeric(ts1$Date_week) %% 2 == 0, 1, 0)
# # Create a new column which indicates additional holidays
# holidayDates <- c("AshWednesday", "GoodFriday", "EasterSunday", "ChristTheKing", "Advent1st", "Advent1st", "Advent3rd", "Advent4th", "ChristmasEve", "ChristmasDay", "BoxingDay", "NewYearsDay","DEAscension", "DECorpusChristi", "DEGermanUnity", "DEChristmasEve", "DENewYearsEve")
# 
# holidays <- NULL
# for (i in holidayDates) {
#   tmp <- holiday(2013:2015, Holiday=i)
#   tmp <- as.character(as.Date(tmp@Data, format=tmp@format))
#   holidays <- c(holidays, tmp)
# }
# holidays <- as.Date(holidays)
# ts1$Holiday <- ifelse(ts1$Date %in% holidays, 1, 0)

# ts1$CompetitionYearMonth <- as.Date(paste(ts1$CompetitionOpenSinceYear, ts1$CompetitionOpenSinceMonth, 1, sep="-"))
# ts1$CompetitionYearMonth_daysdiff <- as.numeric(difftime(time1 = ts1$Date, time2 = ts1$CompetitionYearMonth, units="days"))
# ts1$CompetitionYearMonth_daysdiff[is.na(ts1$CompetitionYearMonth_daysdiff)] <- -9999

# ts1$Promo2SinceWeek[ts1$Promo2SinceWeek==-9999]<-NA
# ts1$Promo2SinceYear[ts1$Promo2SinceYear==-9999]<-NA
# ts1$promoYearWeek <- as.Date(paste(ts1$Promo2SinceYear, ts1$Promo2SinceWeek, 1, sep="-"), format="%Y-%W-%w")
# ts1$promoYearWeek_daysdiff <- as.numeric(difftime(time1 = ts1$Date, time2 = ts1$promoYearWeek, units="days"))
# ts1$promoYearWeek_daysdiff[is.na(ts1$promoYearWeek_daysdiff)] <- -9999

# ts1$Promo2SinceWeek[is.na(ts1$Promo2SinceWeek)] <- -9999
# ts1$Promo2SinceYear[is.na(ts1$Promo2SinceYear)] <- -9999

ts1$Store <- as.factor(ts1$Store)
ts1$DayOfWeek <- as.factor(ts1$DayOfWeek)

ts1$store_dayOfWeek_2way <- cat2WayAvg(ts1, "Store", "DayOfWeek", "Sales", pred0="prob0",filter=ts1$filter==0, k=20, f=10, r_k=0)
ts1$store_mday_2way <- cat2WayAvg(ts1, "Store", "Date_mday", "Sales", pred0="prob0",filter=ts1$filter==0, k=20, f=10, r_k=0)
ts1$store_week_2way <- cat2WayAvg(ts1, "Store", "Date_week", "Sales", pred0="prob0",filter=ts1$filter==0, k=20, f=10, r_k=0)

train <- ts1[ts1$filter==0,]
test <- ts1[ts1$filter==2,]
train <- train[!names(train) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","Promo2YearWeek","promo2YearWeek","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceYear")]
test <- test[!names(test) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","Promo2YearWeek","promo2YearWeek","CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek","Promo2SinceYear")]

addDimTrain <- factorToNumeric(train, train, "Sales", c("Store","DayOfWeek"), metrics = c("mean","median","sd","skewness","kurtosis"))
addDimTest <- factorToNumeric(train, test, "Sales", c("Store","DayOfWeek"), metrics = c("mean","median","sd","skewness","kurtosis"))
train <- cbind(train, addDimTrain)
test <- cbind(test, addDimTest)


train <- train[!names(train) %in% c("Store","Date_mday","Date_week")]
test <- test[!names(test) %in% c("Store","Date_mday","Date_week")]

# ts1 <- ts1[!names(ts1) %in% c("dummy","prob0","Date","CompetitionYearMonth","promoYearWeek")]
# varnames <- c("Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval","Sales", "Date_mon", "Date_mday", "Date_yearmon", "Holiday", "CompetitiontYearMonth_daysdiff", "promoYearWeek_daysdiff")
# trainInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==0, varnames]))[,-1]
# testInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==2, varnames]))[,-1]
# trainInd$Sales <- ts1$Sales[ts1$filter==0]
# testInd$Sales <- ts1$Sales[ts1$filter==2]

# Remove closed stores and those without sales
# Closed and zero sale stores are exluded from scoring
# 0 sale stores in the train set mess up the RMSPE metric -- possible to fix?
train <- train[train$Open==1,]
train <- train[train$Sales!=0,]

train$Open <- NULL

trainInd <- data.frame(model.matrix(Sales ~ ., data=train))[,-1]
testInd <- data.frame(model.matrix(Sales ~ ., data=test))[,-1]
trainInd$Sales <- train$Sales
testInd$Sales <- test$Sales


library(caretEnsemble)
library(RPushbullet)
varnames <- names(trainInd[!names(trainInd) %in% "Sales"])
set.seed(2015)
ensCtrl <- trainControl(method="cv",
                        number=10,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createFolds(train$Sales, k=10, returnTrain=TRUE),
                        summaryFunction=RMSPEsummary)

(tme <- Sys.time())
glmnetCaretEns1 <- train(x = trainInd[,varnames],
                         y = log(trainInd$Sales+1),
                         method="glmnet", 
                         trControl=ensCtrl,
                         metric="RMSPE",
                         maximize=FALSE,
                         tuneGrid=expand.grid(alpha = c(.01,.03,.1,.3),
                                              lambda = c(.01,.03,.1,.3)),
                         preProcess=c("center","scale")
)
(Sys.time() - tme)
save(xgbCaretEns1, file="glmnetCaretEns1.rda")
pbPost(type="note",title="glmnetCaretEns1", body="Done")

foldPreds <- glmnetCaretEns1$pred
foldPreds %>% group_by(Resample) %>% summarise(n=length(Resample), rmspe=sqrt(mean(((exp(pred)-1)/(exp(obs)-1)-1)^2)), minOb=min(exp(obs)-1), maxDiff=max((exp(pred)-1)-(exp(obs)-1)), maxPercDiff = max((exp(pred)-1)/(exp(obs)-1)-1))
