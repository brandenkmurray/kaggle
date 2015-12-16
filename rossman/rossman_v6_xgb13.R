# Create RMPSE metric for training
RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
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

# From Marc in the forums
RMPSE_obj <- function(predts, dtrain) {
  labels <- getinfo(dtrain, "label")
  grad <- ifelse(labels == 0, 0, -1/labels+predts/(labels**2))
  hess <- ifelse(labels == 0, 0, 1/(labels**2))
  return(list(grad = grad, hess = hess))
}


varnames <- names(train[!names(train) %in% "Sales"])

param <- list(eta                 = 0.02, 
              max_depth           = 14,
              subsample           = .9,
              colsample_bytree    = .7
              
)

set.seed(2015)
h <- train$Date_year==2015 & train$Date_mon>=6
dval <- xgb.DMatrix(data=data.matrix(train[h,varnames]),label=log(train$Sales+1)[h])
dtrain <- xgb.DMatrix(data=data.matrix(train[-h,varnames]),label=log(train$Sales+1)[-h])
watchlist <- list(val=dval, train=dtrain)

# dtrain <- xgb.DMatrix(data=data.matrix(train[,varnames]),label=log(train$Sales+1))
set.seed(42069)
tme <- Sys.time()
xgb13 <- xgb.train(params             = param,
                  data                = dtrain,
                  # label             = train$Sales,
                  nrounds             = 6000, 
                  verbose             = 1,
                  print.every.n       = 50,
                  maximize            = FALSE,
                  watchlist           = watchlist,
                  obj                 = RMPSE_obj,
                  feval               = RMPSE,
                  early.stop.round    = 100
)
(runTime <- Sys.time() - tme)
save(xgb13, file="xgb13.rda")

s1 <- s1[order(s1$Store, s1$Date), ]
s1ID <- s1$Id

salesPred <- exp(predict(xgb13, data.matrix(test[,varnames])))-1
salesPred <- salesPred - 100
submission <- data.frame(Id=s1ID, Sales=salesPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-10-25-2015-xgb13_minus200.csv", row.names=FALSE)


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
                      feval               = RMPSE
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
                 metric="RMPSE", 
                 maximize=FALSE)


library(caretEnsemble)
set.seed(2015)
ensCtrl <- trainControl(method="cv",
                        number=3,
                        savePredictions=TRUE,
                        allowParallel=TRUE,
                        index=createFolds(train$Sales, k=3),
                        summaryFunction=RMPSEsummary)
library(doParallel)
cl <- makeCluster(6)
registerDoParallel(cl)
(tme <- Sys.time())
rf1 <- train(x = train[,varnames],
             y = log(train$Sales+1),
             method="rf", 
             trControl=ensCtrl,
             metric="RMPSE",
             maximize=FALSE,
             tuneGrid=expand.grid(mtry=c(10,13)), 
             sampsize=200000,
             nodesize=20, 
             ntree=500)
(Sys.time() - tme)
stopCluster(cl)

tme <- Sys.time()
model_list <- caretList(
  x = train[,varnames],
  y = log(train$Sales+1),
  trControl=ensCtrl,
  metric="RMPSE",
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