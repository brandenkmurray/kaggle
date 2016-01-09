

# Load CV predictions from models
xgb1preds <- read.csv("./stack_models/cvPreds_xgb1.csv")
xgb2preds <- read.csv("./stack_models/cvPreds_xgb2.csv")
xgb3preds <- read.csv("./stack_models/cvPreds_xgb3.csv")
xgb4preds <- read.csv("./stack_models/cvPreds_xgb4.csv")
xgb5preds <- read.csv("./stack_models/cvPreds_xgb5.csv")
kknn1preds <- read.csv("./stack_models/cvPreds_kknn1.csv")
kknn2preds <- read.csv("./stack_models/cvPreds_kknn2.csv")
kknn3preds <- read.csv("./stack_models/cvPreds_kknn3.csv")
kknn4preds <- read.csv("./stack_models/cvPreds_kknn4.csv")
# Edit and bind predictions
xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
xgb3preds$VisitNumber <- NULL
xgb4preds$VisitNumber <- NULL
xgb5preds$VisitNumber <- NULL
kknn1preds$VisitNumber <- NULL
kknn2preds$VisitNumber <- NULL
kknn3preds$VisitNumber <- NULL
kknn4preds$VisitNumber <- NULL
lay1preds <- cbind(xgb1preds, xgb2preds, xgb3preds,xgb4preds,xgb5preds, kknn1preds, kknn2preds, kknn3preds, kknn4preds)

# Add the class column to the dataset
t1 <- data.table(read.csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class)]

lay1preds <- data.table(cbind(class=t1$class, lay1preds))

# Create a validation set
set.seed(1234)
h <- sample(nrow(lay1preds), 2000)
# Create DMatrices
dval <- xgb.DMatrix(data=data.matrix(lay1preds[h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[h,"class", with=FALSE]))
dtrain <- xgb.DMatrix(data=data.matrix(lay1preds[-h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[-h,"class", with=FALSE]))
watchlist <- list(val=dval,train=dtrain)

# Train Model
param <- list(objective="multi:softprob",
              eval_metric="mlogloss",
              num_class=38,
              eta = .05,
              max_depth=3,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=1
)
set.seed(201510)
(tme <- Sys.time())
xgbLay2_v5 <- xgb.train(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgbLay2_v5, file="./stack_models/xgbLay2_v5.rda")


# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read.csv("./stack_models/testPreds_xgb1full.csv")
xgb2fullpreds <- read.csv("./stack_models/testPreds_xgb2full.csv")
xgb3fullpreds <- read.csv("./stack_models/testPreds_xgb3full.csv")
xgb4fullpreds <- read.csv("./stack_models/testPreds_xgb4full.csv")
xgb5fullpreds <- read.csv("./stack_models/testPreds_xgb5full.csv")
kknn1fullpreds <- read.csv("./stack_models/testPreds_kknn1full.csv")
kknn2fullpreds <- read.csv("./stack_models/testPreds_kknn2full.csv")
kknn3fullpreds <- read.csv("./stack_models/testPreds_kknn3full.csv")
kknn4fullpreds <- read.csv("./stack_models/testPreds_kknn4full.csv")
# Edit and bind test set predictions
xgb1fullpreds$VisitNumber <- NULL
xgb2fullpreds$VisitNumber <- NULL
xgb3fullpreds$VisitNumber <- NULL
xgb4fullpreds$VisitNumber <- NULL
xgb5fullpreds$VisitNumber <- NULL
kknn1fullpreds$VisitNumber <- NULL
kknn2fullpreds$VisitNumber <- NULL
kknn3fullpreds$VisitNumber <- NULL
kknn4fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds, xgb3fullpreds, xgb4fullpreds, xgb5fullpreds,kknn1fullpreds, kknn2fullpreds, kknn3fullpreds, kknn4fullpreds)
# Predict the test set using the XGBOOST stacked model
lay2preds <- predict(xgbLay2_v5, newdata=data.matrix(lay1fullpreds))
preds <- data.frame(t(matrix(lay2preds, nrow=38, ncol=length(lay2preds)/38)))
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_v5_preds.csv", row.names=FALSE)
## Score