# Load CV predictions from models
xgb1preds <- read.csv("./stack_models/xgb1_stack_preds.csv")
xgb2preds <- read.csv("./stack_models/xgb2_stack_preds.csv")

# Edit and bind predictions
xgb1preds$VisitNumber <- NULL
xgb2preds$VisitNumber <- NULL
lay1preds <- cbind(xgb1preds, xgb2preds)
# Add the class column to the dataset
lay1preds <- cbind(ts1Trans[filter==0,"class",with=FALSE], lay1preds)

# Create a validation set
h <- sample(nrow(lay1preds), 2000)
# Create DMatrices
dval<-xgb.DMatrix(data=data.matrix(lay1preds[h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[h,"class", with=FALSE]))
dtrain<-xgb.DMatrix(data=data.matrix(lay1preds[-h,2:ncol(lay1preds), with=FALSE]),label=data.matrix(lay1preds[-h,"class", with=FALSE]))
watchlist<-list(val=dval,train=dtrain)

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
xgbLay2 <- xgb.train(data = dtrain,
               params = param,
               nrounds = 6000,
               maximize=FALSE,
               watchlist=watchlist,
               print.every.n = 5,
               early.stop.round=50)
Sys.time() - tme
save(xgb2, file="./stack_models/xgb2.rda")


# Load Test Set predictions from models trained on the entire training set
xgb1fullpreds <- read.csv("./stack_models/xgb1full_stack_preds.csv")
xgb2fullpreds <- read.csv("./stack_models/xgb2full_stack_preds.csv")


# Edit and bind test set predictions
xgb1fullpreds$VisitNumber <- NULL
xgb2fullpreds$VisitNumber <- NULL
lay1fullpreds <- cbind(xgb1fullpreds, xgb2fullpreds)
# Predict the test set using the XGBOOST stacked model
lay2preds <- predict(xgbLay2, newdata=data.matrix(lay1fullpreds))
preds <- data.frame(t(matrix(lay2preds, nrow=38, ncol=length(lay2preds)/38)))
samp <- read.csv('sample_submission.csv')
cnames <- names(samp)[2:ncol(samp)]
names(preds) <- cnames
submission <- data.frame(VisitNumber=samp$VisitNumber, preds)
write.csv(submission, "./stack_models/xgbLay2_preds_test1.csv", row.names=FALSE)
