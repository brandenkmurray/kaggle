library(data.table)
library(caret)
library(dummies)
library(xgboost)
library(data.table)
library(readr)
library(Matrix)
library(irlba)
library(Metrics)
library(MASS)

auc1<-function (actual, predicted) {
  
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}

setwd("/media/branden/SSHD1/kaggle/facebookV")
ts1Trans <- fread("./data_trans/data_trans_v1.csv")



ts1Trans <- ts1Trans[class %in% seq(0,99,1)]
ts1Trans$filter <- c(rep(0,20000), rep(2, nrow(ts1Trans)-20000))


set.seed(201605)
cvFolds <- createFolds(ts1Trans[filter==0,class], k=5, list=FALSE)
classes <- uniqueN(ts1Trans[filter==0,class])
# y <- as.factor(sample(1:100000, 1000000, replace = TRUE))
# y_dum <- sparse.model.matrix(~y, y)

y <- as.factor(ts1Trans[filter==0,class])
y_dum <- sparse.model.matrix(~y-1, y)


# hilbert <- function(n) { i <- 1:n; 1 / outer(i - 1, i, "+") }
# X <- hilbert(1000)[,1:500]
nv=30

V <- irlba(y_dum, nv=nv)$v
# V_norm <- (base::norm(t(V)[,174], type="2")^2)/nv

p_i <- NULL
for (i in 1:classes){
  p_i[i] <- (base::norm(t(V)[,i], type="2")^2)/nv
}

samp <- NULL
set.seed(201605)
while (length(samp) < nv){
  x <- sample(0:(classes-1), 1, replace = TRUE, prob=p_i)
  if(!x %in% samp)
    samp <- c(samp, x)
}
samp <- sort(samp)

Yc <- as.matrix(y_dum[,samp])
Yc_inv <- ginv(Yc)

library(glmnet)
varnames <- c("x","y","accuracy","time")
ts1Trans[,(varnames):=lapply(.SD, scale), .SDcols=varnames]
gc()
input <- sparse.model.matrix(~.-1, ts1Trans[filter==0,varnames,with=FALSE])
testinput <- sparse.model.matrix(~.-1, ts1Trans[filter==2,varnames,with=FALSE])


h <- NULL
for (i in 1:nv){
  labels <- as.numeric(ts1Trans[filter==0, class]==samp[i])
  glmnet1 <- glmnet(x=input,
                      y=as.factor(make.names(labels)),
                      family="binomial",
                      # foldid=cvFolds,
                      # keep=TRUE,
                      maxit=1000000,
                      alpha=.3,
                      lambda=c(0.00000001)
                      )
  
  pred <- predict(glmnet1, newx=testinput, type="response")
  h <- cbind(h, pred)
}

y_hat <- h %*% Yc_inv %*% y_dum
y_hatDF <- as.data.frame(as.matrix(y_hat))
# View(y_hatDF)
# auc1(labels, glmnet1cv$fit.preval[,4])
# logLoss(labels, glmnet1cv$fit.preval[,1])

map3 <- function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = t(matrix(preds, ncol = length(labels)))
  preds = t(apply(preds, 1, order, decreasing = T))[, 1:3] - 1
  succ = (preds == labels)
  w = 1 / (1:3)
  map3 = mean(succ %*% w)
  return (list(metric = 'map3', value = map3))
}

# testPreds <- predict(xgb1, dtest)
# testPreds <- as.data.table(t(matrix(testPreds, nrow=100)))
# classMap <- read_csv("./data_trans/classMap.csv")
colnames(y_hatDF) <- as.character(sort(unique(ts1Trans[filter==0,class])))
# write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb11.csv", row.names=FALSE)
top <- t(apply(y_hatDF, 1, function(y) order(y)[classes:(classes-2)]))
top <- split(top, 1:NROW(top))

mapk(k=3, as.list(ts1Trans[filter==2, class]), top)

testPreds_top5_concat <- do.call("paste", c(testPreds_top5, sep=" "))


