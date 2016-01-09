

# Load CV predictions from models
# xgb1preds <- read.csv("./stack_models/cvPreds_xgb1.csv")
xgb2preds <- read.csv("./stack_models/cvPreds_xgb2.csv")
xgb3preds <- read.csv("./stack_models/cvPreds_xgb3.csv")
xgb7preds <- read.csv("./stack_models/cvPreds_xgb7.csv")
kknn1preds <- read.csv("./stack_models/cvPreds_kknn1.csv")
# Edit and bind predictions
# xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
xgb3preds$VisitNumber <- NULL
xgb7preds$VisitNumber <- NULL
kknn1preds$VisitNumber <- NULL
lay1preds <- cbind(xgb2preds, xgb3preds, xgb7preds, kknn1preds)

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
xgbLay2_v7 <- xgb.train(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgbLay2_v7, file="./stack_models/xgbLay2_v7.rda")


# Load Test Set predictions from models trained on the entire training set

xgb2fullpreds <- read.csv("./stack_models/testPreds_xgb2full.csv")
xgb3fullpreds <- read.csv("./stack_models/testPreds_xgb3full.csv")
xgb7fullpreds <- read.csv("./stack_models/testPreds_xgb7full.csv")
kknn1fullpreds <- read.csv("./stack_models/testPreds_kknn1full.csv")

# Edit and bind test set predictions
xgb2fullpreds$VisitNumber <- NULL
xgb3fullpreds$VisitNumber <- NULL
xgb7fullpreds$VisitNumber <- NULL
kknn1fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb2fullpreds, xgb3fullpreds, xgb7fullpreds, kknn1fullpreds)
# Predict the test set using the XGBOOST stacked model
lay2preds <- predict(xgbLay2_v7, newdata=data.matrix(lay1fullpreds))
preds <- data.frame(t(matrix(lay2preds, nrow=38, ncol=length(lay2preds)/38)))
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_v7_preds.csv", row.names=FALSE)
