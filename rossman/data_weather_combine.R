library(timeDate)
library(zoo)
library(xgboost)
library(sqldf)

# Set working directory
setwd("/home/branden/Documents/kaggle/rossman")
# Import data
t1 <- read.csv("train.csv")
s1 <- read.csv("test.csv")
store <- read.csv("store.csv")
# Merge datasets
t1 <- merge(t1, store)
s1 <- merge(s1, store)

# Save training set Sales separately
t1Sales <- t1$Sales
t1$Sales <- NULL
#Customers only appears in the training set
#Maybe try predicting customers and then relate that to sales?
t1Customers <- t1$Customers
t1$Customers <- NULL
# Create filter column for potential creation of mean value columns
t1$filter <- 0
s1$filter <- 2
t1$Sales <- t1Sales
s1$Sales <- -1
s1ID <- s1$Id
s1$Id <- NULL
# Bind train & test set
ts1 <- rbind(t1, s1)

# Create dummy column for potential creation on mean value columns 
ts1$dummy <- as.factor("A")
ts1$prob0 <- mean(ts1$Sales[ts1$filter==0])
ts1[is.na(ts1)] <- -99


# Separate date elements
ts1$Date <- as.Date(ts1$Date, format="%Y-%m-%d")
ts1$Date_year <- as.integer(strftime(ts1$Date, format="%Y"))
ts1$Date_mon <- strptime(ts1$Date, "%Y-%m-%d")$mon
ts1$Date_mday <- strptime(ts1$Date, "%Y-%m-%d")$mday
ts1$Date_yearmon <- as.numeric(as.yearmon(strptime(ts1$Date, "%Y-%m-%d")))
ts1$Date_week <- as.integer(strftime(ts1$Date, format="%W"))
# Even/odd weeks -- if people are paid biweekly then this could affect purchase patterns
ts1$Date_evenweek <- ifelse(ts1$Date_week %% 2 == 0, 1, 0)

# Load german weather data created via "germanWeather.R" script
germanyWeather <- read.csv("germanyWeather.csv")
germanyWeather$Date <- as.Date(germanyWeather$Date, format="%Y-%m-%d")

# Select the larger cities from the weather dataset
stations <- c("EDAC","EDDB","EDDC","EDDE","EDDF","EDDG","EDDH","EDDI","EDDK","EDDL","EDDM","EDDN","EDDP","EDDS","EDDV","EDDW","EDHI","EDLW","EDVE")
germanyWeather <- germanyWeather[germanyWeather$stationID %in% stations,]
germanyWeather$stationID <- factor(germanyWeather$stationID)
weatherSplit <- split(germanyWeather, germanyWeather$stationID)

weatherFeatures <- c("Max_TemperatureF","Min_TemperatureF","PrecipitationIn","CloudCover","Events")

tme <- Sys.time()
for (i in stations) {
    x <- weatherSplit[[i]][c("Date",weatherFeatures)]
    nColBeg <- dim(ts1)[2] + 1
    ts1 <- sqldf("SELECT a.*, b.Max_TemperatureF, b.Min_TemperatureF, b.PrecipitationIn, b.CloudCover, b.Events FROM ts1 a LEFT JOIN x b ON a.Date==b.Date")
    nColEnd <- nColBeg + length(weatherFeatures) -1
    colnames(ts1)[nColBeg:nColEnd] <- paste(i,weatherFeatures,sep="_")

}
Sys.time() - tme
# Reorder dataset because it was reordered during the merge
ts1 <- ts1[order(ts1$Store, ts1$Date),]

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

eventNames <- grep("Events",names(ts1), value=TRUE)
varnames <- c("StateHoliday","StoreType","Assortment","PromoInterval",eventNames)
for (f in varnames) {
  levels <- unique(ts1[[f]])
  ts1[[f]] <- as.integer(factor(ts1[[f]], levels=levels))
  
}
ts1[is.na(ts1)] <- -99

train <- ts1[ts1$filter==0,]
test <- ts1[ts1$filter==2,]
train <- train[!names(train) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]
test <- test[!names(test) %in% c("dummy","filter","prob0","Date","CompetitionYearMonth","promoYearWeek")]

# ts1 <- ts1[!names(ts1) %in% c("dummy","prob0","Date","CompetitionYearMonth","promoYearWeek")]
# varnames <- c("Store", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval","Sales", "Date_mon", "Date_mday", "Date_yearmon", "Holiday", "CompetitiontYearMonth_daysdiff", "promoYearWeek_daysdiff")
# trainInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==0, varnames]))[,-1]
# testInd <- data.frame(model.matrix(Sales ~ ., data=ts1[ts1$filter==2, varnames]))[,-1]
# trainInd$Sales <- ts1$Sales[ts1$filter==0]
# testInd$Sales <- ts1$Sales[ts1$filter==2]

# Remove closed stores and those without sales
# Closed and zero sale stores are exluded from scoring
# 0 sale stores in the train set mess up the RMPSE metric -- possible to fix?
train <- train[train$Open==1,]
train <- train[train$Sales!=0,]

train$Open <- NULL



