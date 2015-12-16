setwd("/home/branden/Documents/kaggle/libmut")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# keep copy of ID variables for test and train data
train_Id <- train$Id
test_Id <- test$Id



testFrame <- data.frame(group=c("A","A","A","B","B","B","B","C","C","C","C","C"), val=c(1,0,1,0,0,0,1,0,1,1,1,0))
str(testFrame)

temp <- data.frame(c(rep(0,nrow(testFrame))), row.names = NULL)
x <- tapply(testFrame[, "val"], testFrame[, "group"], mean)
x <- data.frame(row.names(x),x, row.names = NULL)
temp <- data.frame(temp,round(lookup(test[,variable], x),2))
colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")


temp <- data.frame(c(rep(0,nrow(testFrame))), row.names = NULL)
for (i in 1:nrow(testFrame)){
  testTemp <- testFrame[-i,]
  temp[i,] <- mean(testTemp[testTemp$group==testFrame[i,"group"], "val"])
}

factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)
  
  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}





leaveOneOutEncode <- function(train, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(train))), row.names = NULL)
  
  for (variable in variables){
    for (metric in metrics){
      for (i in 1:nrow(train)){
        factLev <- as.character(train[i, variable])
        testTemp <- train[-i,]
        testTemp <- testTemp[testTemp[variable]==factLev,]
        temp[i,] <- apply(as.matrix(testTemp[response]), 2, metric) * rnorm(1, mean=1, sd=.02)
      }
  }
 
  }
  return (temp) 
}

tme <- Sys.time()
temp <- data.frame(c(rep(0,nrow(train[1:1000,]))), row.names = NULL)
for (i in 1:nrow(train[1:1000,])){
  factLev <- as.character(train[i, "T2_V13"])
  testTemp <- train[-i,c("T2_V13","Hazard")]
  testTemp <- testTemp[testTemp["T2_V13"]==factLev,]
  temp[i,] <- apply(as.matrix(testTemp["Hazard"]), 2, mean) * rnorm(1, mean=1, sd=.02)
}
Sys.time() - tme






