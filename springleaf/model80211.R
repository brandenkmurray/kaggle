require(readr)
require(pROC)
library(xgboost)
require(cvTools)
require(randomForest)
library(RPushbullet)
library(readr)
# require(sprint)


cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")
gc()

set.seed(1)


#####From xgboost_submission_with_cv_2.csv

feature.names <- names(train)[2:(ncol(train)-1)]

cat("transform non-numeric fields to factors\n")
cat("Detect datetime field and transform them to numeric")
train.fix <- train[,feature.names]
test.fix <- test[,feature.names]
for (f in feature.names) {
  if((sum(!is.na(train[[f]]))+sum(!is.na(test[[f]]))) == 0){
    #### In case the who column is NA, na.roughfix doesnt know how to deal with
    train.fix[[f]] <- 0
    test.fix[[f]] <- 0
    
  }else  if (class(train[[f]])=="character" |class(train[[f]])=="factor" | (class(train[[f]])=="logical")){
    levels <- unique(c(train[[f]], test[[f]]))
    if((length(levels)) < 100){
      train.fix[[f]] <- as.factor(factor(train[[f]], levels=levels))
      test.fix[[f]]  <- as.factor(factor(test[[f]],  levels=levels))
    }else{
      train.fix[[f]] <- as.numeric(factor(train[[f]], levels=levels))
      test.fix[[f]]  <- as.numeric(factor(test[[f]],  levels=levels))
    } 
    tmp <- train[[f]]
    if(!is.na(strptime(max(tmp,na.rm=TRUE),"%d%h%y:%H:%M:%S"))){
      cat(f)
      cat("\n")
      train.fix[[paste0(f,"_num")]] <- as.numeric(strptime(train[[f]], "%d%h%y:%H:%M:%S"))
      test.fix[[paste0(f,"_num")]] <- as.numeric(strptime(test[[f]], "%d%h%y:%H:%M:%S"))
      train.fix[[paste0(f,"_wkday")]]  <- strptime(train[[f]], "%d%h%y:%H:%M:%S")$wday
      test.fix[[paste0(f,"_wkday")]]  <- strptime(test[[f]], "%d%h%y:%H:%M:%S")$wday
      train.fix[[paste0(f,"_mon")]]  <- strptime(train[[f]], "%d%h%y:%H:%M:%S")$mon
      test.fix[[paste0(f,"_mon")]]  <- strptime(test[[f]], "%d%h%y:%H:%M:%S")$mon
      train.fix[[paste0(f,"_mday")]]  <- strptime(train[[f]], "%d%h%y:%H:%M:%S")$mday
      test.fix[[paste0(f,"_mday")]]  <- strptime(test[[f]], "%d%h%y:%H:%M:%S")$mday
      train.fix[[paste0(f,"_year")]]  <- strptime(train[[f]], "%d%h%y:%H:%M:%S")$year
      test.fix[[paste0(f,"_year")]]  <- strptime(test[[f]], "%d%h%y:%H:%M:%S")$year
      train.fix[[paste0(f,"_hour")]]  <- strptime(train[[f]], "%d%h%y:%H:%M:%S")$hour
      test.fix[[paste0(f,"_hour")]]  <- strptime(test[[f]], "%d%h%y:%H:%M:%S")$hour
      
    }
  }
  else{
    train.fix[[f]] <- train[[f]]
    test.fix[[f]] <- test[[f]]
  }
}


all.fix <- rbind(train.fix, test.fix)

all.fix <- na.roughfix(all.fix)

train.fix <- all.fix[1:nrow(train),]
test.fix <- all.fix[(nrow(train)+1):(nrow(train)+nrow(test)),]

train.fix$target <- train$target
test.fix$target <- test$target

feature.names.fix <- names(train.fix)[1:(ncol(train)-1)]

param <- list( eta                 = 0.005,
               max_depth           = 16,  # changed from default of 8
               subsample           = 0.8,
               colsample_bytree    = 0.8
               
)


tme <- Sys.time()
xgb9 <- xgboost(data        = data.matrix(train.fix[,feature.names.fix]),
               label       = train.fix$target,
               params = param,
               nrounds     = 20000,
               objective   = "binary:logistic",
               eval_metric = "auc")
(runTime <- Sys.time() - tme)
pbPost(type = "note", title = "XGB9", body="Done.")
save(xgb9, file="xgb9.rda")

targetPred <- predict(xgb9, data.matrix(test.fix[,feature.names.fix]))
submission <- data.frame(ID=test$ID, target=targetPred)
colnames(submission)[2] <- "target"
write.csv(submission, "submission-09-24-2015-xgb9-allVars.csv", row.names=FALSE)
