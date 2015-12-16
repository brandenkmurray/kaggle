library(timeDate)
library(zoo)
library(xgboost)
setwd("/home/branden/Documents/kaggle/rossman")
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
ts1[is.na(ts1)] <- -99

varnames <- c("StateHoliday","StoreType","Assortment","PromoInterval")
for (f in varnames) {
    levels <- unique(ts1[[f]])
    ts1[[f]] <- as.integer(factor(ts1[[f]], levels=levels))
  
}


# Separate date elements
ts1$Date <- as.Date(ts1$Date, format="%Y-%m-%d")
ts1$Date_numeric <- as.numeric(ts1$Date)
ts1$Date_year <- as.integer(strftime(ts1$Date, format="%Y"))
ts1$Date_mon <- strptime(ts1$Date, "%Y-%m-%d")$mon
ts1$Date_mday <- strptime(ts1$Date, "%Y-%m-%d")$mday
ts1$Date_yearmon <- as.numeric(as.yearmon(strptime(ts1$Date, "%Y-%m-%d")))
ts1$Date_week <- as.integer(strftime(ts1$Date, format="%W"))
# Even/odd weeks -- if people are paid biweekly then this could affect purchase patterns
ts1$Date_evenweek <- ifelse(ts1$Date_week %% 2 == 0, 1, 0)
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
# 0 sale stores in the train set mess up the RMSPE metric -- possible to fix?
train <- train[train$Open==1,]
train <- train[train$Sales!=0,]

train$Open <- NULL

# Create RMSPE metric for training
RMSPE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMSPE", value = err))
}

RMSPEsummary <- function (data,
                          lev = NULL,
                          model = NULL) {
  elab<-exp(as.numeric(data[,"obs"]))-1
  epreds<-exp(as.numeric(data[,"pred"]))-1
  out <- sqrt(mean((epreds/elab-1)^2))
  names(out) <- "RMSPE"
  out
}


# objective function from Marc in the forums
RMSPE_objective <- function(predts, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- ifelse(labels == 0, 0, -1/labels+predts/(labels**2))
  hess <- ifelse(labels == 0, 0, 1/(labels**2))
  return(list(grad = grad, hess = hess))
}


varnames <- names(train[!names(train) %in% "Sales"])

param <- list(objective           = RMSPE_objective, 
              eta                 = 0.02, 
              max_depth           = 10,
              subsample           = .9,
              colsample_bytree    = .7
            
)

h <- sample(nrow(train),80000)
dval <- xgb.DMatrix(data=data.matrix(train[h,varnames]),label=log(train$Sales+1)[h])
dtrain <- xgb.DMatrix(data=data.matrix(train[-h,varnames]),label=log(train$Sales+1)[-h])
watchlist <- list(val=dval, train=dtrain)

dtrain <- xgb.DMatrix(data=data.matrix(train[,varnames]),label=log(train$Sales+1))
set.seed(42069)
tme <- Sys.time()
xgb7 <- xgb.train(params              = param,
                  data        = dtrain,
                 # label       = train$Sales,
                 nrounds             = 5000, 
                verbose             = 0,
                print.every.n       = 50,
                maximize            = FALSE,
                watchlist = watchlist,
                feval               = RMSPE
)
(runTime <- Sys.time() - tme)
save(xgb7, file="xgb7.rda")

salesPred <- exp(predict(xgb7, data.matrix(test[,varnames])))-1
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-17-2015-xgb7.csv", row.names=FALSE)


h <- sample(nrow(train),10000)
dval <- xgb.DMatrix(data=data.matrix(train[h,varnames]),label=log(train$Sales+1)[h])
dtrain <- xgb.DMatrix(data=data.matrix(train[-h,varnames]),label=log(train$Sales+1)[-h])
watchlist <- list(val=dval, train=dtrain)


grid <- expand.grid(eta=c(0.03, .01), max_depth=c(8,10,12,14), nrounds=c(15000))
# grid <- expand.grid(eta=c(0.1), max_depth=c(6), nrounds=c(100))
predFrame <- as.data.frame(matrix(ncol=nrow(grid), nrow=nrow(testInd)))
set.seed(42069)
tme <- Sys.time()
for (i in 1:nrow(grid)){
      print(paste("Building Model",i))
      seed <- 42069 + i
      set.seed(seed)
      
      # h <- sample(nrow(train),10000)
      # dval <- xgb.DMatrix(data=data.matrix(train[h,varnames]),label=log(train$Sales+1)[h])
      dtrain <- xgb.DMatrix(data=data.matrix(train[,varnames]),label=log(train$Sales+1))
      # watchlist <- list(val=dval, train=dtrain)
      
      param <- list(objective           = "reg:linear", 
                    eta                 = grid[i,"eta"], 
                    max_depth           = grid[i,"max_depth"],
                    subsample           = 0.7,
                    colsample_bytree    = 0.7
                    
      ) 
      
      xgbtmp <- xgb.train(params            = param,
                        data                = dtrain,
                        # label             = train$Sales,
                        verbose             = 1,
                        nrounds             = grid[i,"nrounds"],
                        print.every.n       = 100,
                        # early.stop.round    = 50,
                        # watchlist           = watchlist,
                        maximize            = FALSE,
                        feval               = RMSPE
      )
      
      predtmp <- exp(predict(xgbtmp, data.matrix(test[,varnames])))-1
      predFrame[i] <- predtmp
      colnames(predFrame)[i] <- paste0("eta",grid[i,"eta"],"dep",grid[i,"max_depth"],"rnd",grid[i,"nrounds"])
      rm(xgbtmp)
      gc()
}
(runTime <- Sys.time() - tme)
write.csv(predFrame, "predFrame-xgbLoop8.csv")


salesPred <- rowMeans(predFrame)
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-09-2015-xgbLoop8.csv", row.names=FALSE)

glmnet2 <- train(x=train[,varnames], 
                 y=log(train$Sales+1), 
                 method="glmnet", 
                 trControl=ensCtrl, 
                 tuneGrid=expand.grid(alpha=c(.001,.003,.01), lambda=c(.003,.01)), 
                 metric="RMSPE", 
                 maximize=FALSE)


# The plan is to build the caret models separately, but keeping the same folds
# Put models into a list, rename the list class as "CaretList", then build ensemble models
library(caretEnsemble)
library(RPushbullet)
varnames <- names(train[!names(train) %in% "Sales"])
set.seed(2015)
ensCtrl <- trainControl(method="cv",
                        number=10,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createFolds(train$Sales, k=10, returnTrain=TRUE),
                        summaryFunction=RMSPEsummary)

# XGBTree Caret Model #1
# Fix the caret xgbTree method so it uses RMSPE_objective as the objective
xgbTreeNew <- getModelInfo(model = "xgbTree", regex = FALSE)[[1]]
xgbTreeNew$fit <- function(x, y, wts, param, lev, last, classProbs, ...) { 
  if(is.factor(y)) {
    if(length(lev) == 2) {
      y <- ifelse(y == lev[1], 1, 0) 
      dat <- xgb.DMatrix(as.matrix(x), label = y)
      out <- xgb.train(list(eta = param$eta, 
                            max_depth = param$max_depth), 
                       data = dat,
                       nrounds = param$nrounds,
                       objective = "binary:logistic",
                       ...)
    } else {
      y <- as.numeric(y) - 1
      dat <- xgb.DMatrix(as.matrix(x), label = y)
      out <- xgb.train(list(eta = param$eta, 
                            max_depth = param$max_depth), 
                       data = dat,
                       num_class = length(lev),
                       nrounds = param$nrounds,
                       objective = "multi:softprob",
                       ...)
    }     
  } else {
    dat <- xgb.DMatrix(as.matrix(x), label = y)
    out <- xgb.train(list(eta = param$eta, 
                          max_depth = param$max_depth), 
                     data = dat,
                     nrounds = param$nrounds,
                     objective = RMSPE_objective,
                     ...)
  }
  out
}

(tme <- Sys.time())
xgbCaretEns1 <- train(
             x = train[,varnames],
             y = log(train$Sales+1),
             method=xgbTreeNew, 
             trControl=ensCtrl,
             metric="RMSPE",
             maximize=FALSE,
             tuneGrid=expand.grid(max_depth = c(10),
                                  nrounds = c(4000),
                                  eta = c(.02)),
             min_child_weight=1,
             subsample=.9,
             colsample_bytree=.7,
             feval=RMSPE)
(Sys.time() - tme)
save(xgbCaretEns1, file="xgbCaretEns1.rda")
pbPost(type="note",title="xgbCaretEns1", body="Done")

foldPreds <- xgbCaretEns1$pred
foldPreds %>% group_by(Resample) %>% summarise(n=length(Resample), rmspe=sqrt(mean(((exp(pred)-1)/(exp(obs)-1)-1)^2)), minOb=min(exp(obs)-1), maxDiff=max((exp(pred)-1)-(exp(obs)-1)), maxPercDiff = max((exp(pred)-1)/(exp(obs)-1)-1))

## GLMNET Caret Model
(tme <- Sys.time())
glmnetCaretEns1 <- train(x = train[,varnames],
                      y = log(train$Sales+1),
                      method="glmnet", 
                      trControl=ensCtrl,
                      metric="RMSPE",
                      maximize=FALSE,
                      tuneGrid=expand.grid(alpha = c(.01,.05,.1,.3),
                                           lambda = c(.01,.05,.1,.3))
                      )
(Sys.time() - tme)
save(xgbCaretEns1, file="xgbCaretEns1.rda")
pbPost(type="note",title="xgbCaretEns1", body="Done")

tme <- Sys.time()
model_list <- caretList(
  x = train[,varnames],
  y = log(train$Sales+1),
  trControl=ensCtrl,
  metric="RMSPE",
  maximize=FALSE,
  tuneList=list(
    rf1=caretModelSpec(method="rf", 
                       tuneGrid=expand.grid(mtry=c(3,5,7,10)), 
                       nodesize=5, 
                       ntree=2000),
    xgb1=caretModelSpec(method="xgbTree", 
                        tuneGrid=expand.grid(max_depth = c(10),
                                             nrounds = c(4000),
                                             eta = c(.02)),
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=1)
    
  )
)
stopCluster(cl)
Sys.time() - tme
pbPost("note", "Ensemble", "Finished.")