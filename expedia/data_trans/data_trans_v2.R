library(data.table)
library(zoo)
library(caret)
library(e1071)
library(Matrix)
# library(proxy)
# library(qlcMatrix)
# library(cccd)
# library(igraph)
library(gtools)
# library(plyr)
# library(dplyr)
# library(sqldf)
# library(DMwR)
# library(Rtsne)
library(doParallel)
library(doRNG)
# library(WGCNA)
# library(VGAM)
# library(Boruta)
library(readr)
library(RcppRoll)
library(Metrics)
# setwd("/media/branden/SSHD1/kaggle/expedia")
# setwd("/media/branden/SSHD1/kaggle/expedia")
setwd("~/ebs")
# load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
source("./data_trans/utils.R")

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#ndcg metric from air's script
# map5 <- function(preds, dtrain) {
#   labels <- as.list(getinfo(dtrain,"label"))
#   num.class = 100
#   pred <- matrix(preds, nrow = num.class)
#   top <- t(apply(pred, 2, function(y) order(y)[num.class:(num.class-4)]-1))
#   top <- split(top, 1:NROW(top))
#   
#   map <- mapk(5, labels, top)
#   return(list(metric = "map5", value = map))
# }

map5 <- function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  preds = t(matrix(preds, ncol = length(labels)))
  preds = t(apply(preds, 1, order, decreasing = T))[, 1:5] - 1
  succ = (preds == labels)
  w = 1 / (1:5)
  map5 = mean(succ %*% w)
  return (list(metric = 'map5', value = map5))
}
#######################
## Load data
#######################
t1 <- fread("./train.csv")
s1 <- fread("./test.csv")


t1$id <- -1
s1 <- s1[,c("is_booking"):=1][,c("cnt", "hotel_cluster"):=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(id2=seq(0,nrow(ts1)-1,1),pred0=Mode(t1$hotel_cluster), dummy="A", filter=c(rep(0,31400000), rep(1,nrow(t1)-31400000), rep(2, nrow(s1))), ts1)


rm(t1, s1)
dates <- c("srch_ci","srch_co")
ts1[,c("date_time"):=lapply(.SD,function(x) as.POSIXct(x, tz="GMT", format="%Y-%m-%d %H:%M:%S")), .SDcols=c("date_time")][,(dates):=lapply(.SD, function(x) as.Date(x, tz="GMT", format="%Y-%m-%d")), .SDcols=dates][,date:=as.Date(date_time, tz="GMT", format="%Y-%m-%d")]

ts1 <- ts1[,':='(date_time_hour=hour(date_time),
                 date_time_wday=wday(date_time),
                 date_time_week=week(date_time),
                 date_time_month=month(date_time),
                 date_time_mday=mday(date_time),
                 date_time_yday=yday(date_time),
                 date_time_quarter=quarter(date_time),
                 srch_ci_mday=mday(srch_ci),
                 srch_ci_yday=yday(srch_ci),
                 srch_co_mday=mday(srch_co),
                 srch_co_yday=yday(srch_co)
                 
)]



ts1[,":="(dist_clust_bookings_sum=sum(is_booking[filter==0])-c(is_booking[filter==0],rep(0, length(is_booking[filter %in% c(1,2)]))),
          dist_clust_cnts_sum=sum(cnt[filter==0])-c(cnt[filter==0],rep(0, length(cnt[filter %in% c(1,2)]))),
          dist_clust_bookings_count=length(is_booking[filter==0])),
    keyby=list(orig_destination_distance)]
ts1[,":="(destid_clust_bookings_sum=sum(is_booking[filter==0])-c(is_booking[filter==0],rep(0, length(is_booking[filter %in% c(1,2)]))),
          destid_clust_cnts_sum=sum(cnt[filter==0])-c(cnt[filter==0],rep(0, length(cnt[filter %in% c(1,2)]))),
          destid_clust_bookings_count=length(is_booking[filter==0])),
    keyby=list(srch_destination_id)]


# # Convert check-in and check-out dates beyond 2017 as NA
# ts1$srch_ci[ts1$srch_ci_year >= 2018] <- NA
# ts1$srch_co[ts1$srch_co_year >= 2018] <- NA
# # Calculate the cumulative number of times a hotel was booked
# # Calculate the number of times a hotel was booked in the previous X days, weeks, months, year
# 
# # ts2 <- copy(ts1)
# vars <- grep("bookings_srch_ci",colnames(ts1),value = TRUE)
# ts1[, (vars) := lapply(vars, function(x) {
#   x <- get(x)
#   x[is.na(x)] <- 0
#   x
# })]


# leak <- fread("./data_trans/leak_props.csv")
# 
# ts1 <- merge(ts1, leak, by=c("user_location_country","user_location_region","user_location_city","hotel_market","orig_destination_distance"), all.x=TRUE, sort=FALSE)
# rm(leak)
dates <- c("date_time", "srch_ci","srch_co","date")
ts1[,(dates):=NULL]
for (j in 2:ncol(ts1)){
  set(ts1, which(is.na(ts1[[j]])),j,-1)}

fwrite(ts1, "./data_trans/ts1_v1.csv")






# dest <- fread("./destinations.csv")
# dest_pca <- preProcess(dest[,2:ncol(dest),with=FALSE], method = "pca", thresh=0.9)
# dest_new <- predict(dest_pca, dest)
# for (col in 2:ncol(dest_new)) {
#   set(dest_new, j=col+0L, value=as.integer(dest_new[[col+0L]]*1000000))
# }
# rm(dest)
dest_pca <- fread("./data_trans/data_trans_pca.csv", key="id2")
ts1 <- fread("./data_trans/ts1_v1.csv", key="id2") 
ts1[dest_pca, (colnames(dest_pca)[2:ncol(dest_pca)]):=dest_pca[,2:ncol(dest_pca),with=FALSE]]; rm(dest_pca); gc()
# ts1 <- merge(ts1, dest_pca, all.x=TRUE, sort=FALSE);rm(dest_pca)
# setkey(ts1, id2)
# ts1 <- ts1[,1:61,with=FALSE]
# fwrite(ts1, "./data_trans/ts1_v1.csv")
# for (j in seq_len(ncol(ts1)))
#   set(ts1,which(is.na(ts1[[j]])),j,-999)
leak <- fread("./data_trans/data_trans_leak_props.csv", key="id2")
ts1[leak, (colnames(leak)[2:ncol(leak)]):=leak[,2:ncol(leak),with=FALSE]]; rm(leak); gc()
destid_clust <- fread("./data_trans/data_trans_destid_wtdprops.csv", key="id2")
ts1[destid_clust, (colnames(destid_clust)[2:ncol(destid_clust)]):=destid_clust[,2:ncol(destid_clust),with=FALSE]]; rm(destid_clust); gc()
destid_datePop <- fread("./data_trans/data_trans_destid_datePop.csv", key="id2")
ts1[destid_datePop, (colnames(destid_datePop)[4:ncol(destid_datePop)]):=destid_datePop[,4:ncol(destid_datePop),with=F]]; rm(destid_datePop);gc()
# ts1 <- ts1[destid_datePop[,3:ncol(destid_datePop),with=FALSE]]; rm(destid_datePop); gc()
leak_datePop <- fread("./data_trans/data_trans_leak_datePop.csv", key="id2")
ts1[leak_datePop, (grep("Market",colnames(leak_datePop), value=TRUE)):=leak_datePop [,grep("Market",colnames(leak_datePop), value=TRUE),with=F]]; rm(leak_datePop);gc()
# ts1 <- ts1[leak_datePop[,c("id2",grep("Market",colnames(leak_datePop), value=TRUE)),with=FALSE]]; rm(leak_datePop);gc()
destid_srchciPop <- fread("./data_trans/data_trans_destid_srchciPop.csv", key="id2")
ts1[destid_srchciPop, (colnames(destid_srchciPop)[3:ncol(destid_srchciPop)]):=destid_srchciPop[,3:ncol(destid_srchciPop),with=F]]; rm(destid_srchciPop);gc()
# ts1 <- ts1[destid_srchciPop[,2:ncol(destid_srchciPop),with=FALSE]]; rm(destid_srchciPop); gc()

fwrite(ts1, "./data_trans/ts1_merged_v1.csv")


ts1 <-  fread("./data_trans/ts1_merged_v1.csv")
# ts1 <- cbind(ts1, leak[,2:ncol(leak),with=FALSE], destid_datePop[,4:ncol(destid_datePop),with=FALSE], leak_datePop, destid_srchciPop)
# rm(leak, destid_datePop, leak_datePop, destid_srchciPop);
# ts1[is.na(ts1)] <- -1

varnames <- colnames(ts1)[!colnames(ts1) %in% c("cnt","id","id2","hotel_cluster","filter","dummy","pred0","date_time", "srch_ci","srch_co","date")]
set.seed(2018)
samp <- sample(1:31400000, 1000000, replace=FALSE)
# valrows<-nrow(ts1[-(1:31400000)][filter==0 & is_booking==1])
# set.seed(2018)
# valsamp <- sample(1:valrows, 500000, replace=FALSE)
library(xgboost)
dtrain <- xgb.DMatrix(data=data.matrix(ts1[filter==0,varnames, with=FALSE][samp]),label=data.matrix(ts1[filter==0, hotel_cluster][samp]))
dval <- xgb.DMatrix(data=data.matrix(ts1[filter==1 & is_booking==1,varnames, with=FALSE]),label=data.matrix(ts1[filter==1 & is_booking==1, hotel_cluster]))
dtest <- xgb.DMatrix(data=data.matrix(ts1[filter==2,varnames, with=FALSE]))
# cvFoldsList <-createFolds(ts1[filter==0, hotel_cluster][samp], k=3)
watchlist <- list(dval=dval, dtrain=dtrain)
# rm(ts1)
gc()




param <- list(objective="multi:softprob",
              eval_metric=map5,
              num_class=100,
              eta = .1,
              max_depth=5,
              min_child_weight=1,
              subsample=.5,
              colsample_bytree=.3,
              nthread=threads)


(tme <- Sys.time())
set.seed(201512)
xgb1cv <- xgb.train(data = dtrain,
                    params = param,
                    nrounds = 400,
                    # folds=cvFoldsList,
                    maximize=TRUE,
                    # prediction=TRUE,
                    watchlist=watchlist,
                    print.every.n = 10,
                    early.stop.round=15)
Sys.time() - tme
save(xgb1cv, file="./stack_models/xgb1cv.rda")



cvPreds <- xgb1cv$pred
cnames <- paste("xgb1", 0:99, sep="_")
colnames(cvPreds) <- cnames
write.csv(data.frame(id=ts1[filter==0,"id",with=FALSE][samp], cvPreds), "./stack_models/cvPreds_xgb1.csv", row.names=FALSE) 


rounds <- floor(which.max(xgb1cv$dt$test.map5.mean) * 1.0)

(tme <- Sys.time())
set.seed(201512)
xgb1 <- xgb.train(data = dtrain,
                  params = param,
                  nrounds = rounds,
                  maximize=TRUE,
                  print.every.n = 20)
Sys.time() - tme
save(xgb1, file="./stack_models/xgb1.rda")

valPreds <- predict(xgb1cv, dval)
valPreds <- as.data.table(t(matrix(valPreds, nrow=100)))
# classMap <- read_csv("./data_trans/classMap.csv")
colnames(valPreds) <- as.character(0:99)
valPreds_top5 <- as.data.frame(t(apply(valPreds, 1, function(x) names(sort(x, decreasing=T)[1:5]))))


testPreds <- predict(xgb1cv, dtest)
testPreds <- as.data.table(t(matrix(testPreds, nrow=100)))
# classMap <- read_csv("./data_trans/classMap.csv")
colnames(testPreds) <- as.character(0:99)
# write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb11.csv", row.names=FALSE)
testPreds_top5 <- as.data.frame(t(apply(testPreds, 1, function(x) names(sort(x, decreasing=T)[1:5]))))

testPreds_top5_concat <- do.call("paste", c(testPreds_top5, sep=" "))

submission <- data.table(id=seq(0,nrow(testPreds_top5)-1,1), hotel_cluster=testPreds_top5_concat)
fwrite(submission, "./stack_models/L1/testPreds/L1_testPreds_xgb1_v13.csv")
xgb1Imp <- xgb.importance(feature_names=varnames, model=xgb1cv)
View(xgb1Imp)




L1_testPreds_xgb1 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1.csv")




leak <- leak[ts1$filter==2]
leak_rowSums <- rowSums(leak[,2:ncol(leak),with=F])
leak_rows <- leak_rowSums>=0

colnames(leak) <- c("id2",as.character(0:99))
# write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb11.csv", row.names=FALSE)
testPreds_top5_leak <- as.data.frame(t(apply(leak[,2:ncol(leak),with=F], 1, function(x) names(sort(x, decreasing=T)[1:5]))))

testPreds_top5_concat <- do.call("paste", c(testPreds_top5_leak, sep=" "))
testPreds_top5_leak_frame <- data.table(id2=leak$id2, hotel_cluster=testPreds_top5_concat)


L1_testPreds_xgb1$hotel_cluster[leak_rows] <- testPreds_top5_leak_frame$hotel_cluster[leak_rows]
fwrite(L1_testPreds_xgb1, "./stack_models/L1/testPreds/L1_testPreds_xgb1_leaktest.csv")
# create submission 
# ids <- NULL
# for (i in 1:NROW(ts1Trans[filter==2,])) {
#   idx <- as.character(ts1Trans$id[ts1Trans$filter==2][i])
#   ids <- append(ids, rep(idx,5))
# }
idx = ts1Trans$id[ts1Trans$filter==2]
id_mtx <- matrix(idx, 1)[rep(1,5), ]
ids <- c(id_mtx)
submission <- NULL
submission$id <- ids
submission$country <- testPreds_top5

# generate submission file
submission <- as.data.frame(submission)
write.csv(submission, "./stack_models/testPreds_xgb19.csv", quote=FALSE, row.names = FALSE)



