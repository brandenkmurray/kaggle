library(readr)
library(data.table)
library(zoo)
library(caret)
library(e1071)
library(Matrix)
library(proxy)
library(qlcMatrix)
library(cccd)
library(igraph)
library(gtools)
library(plyr)
library(dplyr)
library(sqldf)
library(DMwR)
library(Rtsne)
library(doParallel)
library(doRNG)
setwd("/media/branden/SSHD1/kaggle/bnp")
load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
# Function from Owen's Amazon competition (https://github.com/owenzhang/Kaggle-AmazonChallenge2013/blob/master/__final_utils.R)
#2 way count
my.f2cnt<-function(th2, vn1, vn2, filter=TRUE) {
  df<-data.frame(f1=th2[,vn1,with=FALSE], f2=th2[,vn2,with=FALSE], filter=filter)
  colnames(df) <- c("f1", "f2", "filter")
  sum1<-sqldf("select f1, f2, count(*) as cnt from df where filter=1 group by 1,2")
  tmp<-sqldf("select b.cnt from df a left join sum1 b on a.f1=b.f1 and a.f2=b.f2")
  tmp$cnt[is.na(tmp$cnt)]<-0
  return(tmp$cnt)
}

my.f3cnt <- function(th2, vn1, vn2, vn3, filter=TRUE) {
  df<-data.frame(f1=th2[,vn1, with=FALSE], f2=th2[,vn2, with=FALSE], f3=th2[, vn3, with=FALSE], filter=filter)
  colnames(df) <- c("f1", "f2", "f3", "filter")
  sum1<-sqldf("select f1, f2, f3, count(*) as cnt from df where filter=1 group by 1, 2, 3")
  tmp<-sqldf("select b.cnt from df a left join sum1 b on a.f1=b.f1 and a.f2=b.f2 and a.f3=b.f3")
  tmp$cnt[is.na(tmp$cnt)]<-0
  return(tmp$cnt)
}

int3WayBool <- function(th2, vn1, vn2, vn3, filter=TRUE) {
  df<-data.frame(f1=th2[,vn1, with=FALSE], f2=th2[,vn2, with=FALSE], f3=th2[, vn3, with=FALSE], filter=filter)
  colnames(df) <- c("f1", "f2", "f3", "filter")
  tmp <- ifelse(df$f1>0 & df$f2>0 & df$f3>0, apply(df[,c("f1","f2","f3")], MARGIN = 1, sum, na.rm=TRUE), 0)
  return(tmp)
}

cat2WayAvg <- function(data, var1, var2, y, pred0, filter, k, f, lambda=NULL, r_k){
  sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c("v1","v2","y","pred0","filt")
  sum1 <- sqldf("SELECT v1, v2, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 WHERE filt=1 GROUP BY v1, v2")
  tmp1 <- sqldf("SELECT b.v1, b.v2, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub1 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2 ")
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)]<-0
  tmp1$cnt1 <- tmp1$cnt
  tmp1$cnt1[filter] <- tmp1$cnt[filter] - 1
  tmp1$sumy1 <- tmp1$sumy
  tmp1$sumy1[filter] <- tmp1$sumy[filter] - tmp1$y[filter]
  tmp1$avgp <- with(tmp1, sumy1/cnt1)
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt1 - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgp + tmp1$beta*tmp1$pred0
  tmp1$avgp[is.na(tmp1$avgp)] <- tmp1$pred0[is.na(tmp1$avgp)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  tmp1$adj_avg[filter]<-tmp1$adj_avg[filter]*(1+(runif(sum(filter))-0.5)*r_k)
  return(tmp1$adj_avg)
}

cat2WayAvgCV <- function(data, var1, var2, y, pred0, filter, k, f, lambda=NULL, r_k, cv=NULL){
  # It is probably best to sort your dataset first by filter and then by ID (or index)
  ind <- unlist(cv, use.names=FALSE)
  oof <- NULL
  if (length(cv) > 0){
    for (i in 1:length(cv)){
      sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
      sub1 <- sub1[sub1$filt==TRUE,]
      sub1$filt <- NULL
      colnames(sub1) <- c("v1","v2","y","pred0")
      sub2 <- sub1[cv[[i]],]
      sub1 <- sub1[-cv[[i]],]
      sum1 <- sqldf("SELECT v1, v2, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 GROUP BY v1, v2")
      tmp1 <- sqldf("SELECT b.v1, b.v2, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub2 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2")
      tmp1$cnt[is.na(tmp1$cnt)] <- 0
      tmp1$sumy[is.na(tmp1$sumy)] <- 0
      if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt - k)/f))
      tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
      tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
      tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
      tmp1$adj_avg <- tmp1$adj_avg*(1+(runif(nrow(sub2))-0.5)*r_k)
      oof <- c(oof, tmp1$adj_avg)
    }
  }
  oofInd <- data.frame(ind, oof)
  oofInd <- oofInd[order(oofInd$ind),]
  sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c("v1","v2","y","pred0","filt")
  sub2 <- sub1[sub1$filt==F,]
  sub1 <- sub1[sub1$filt==T,]
  sum1 <- sqldf("SELECT v1, v2, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 GROUP BY v1, v2")
  tmp1 <- sqldf("SELECT b.v1, b.v2, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub2 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2")
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)] <- 0
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
  tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  # Combine train and test into one vector
  return(c(oofInd$oof, tmp1$adj_avg))
}

cat3WayAvg <- function(data, var1, var2, var3, y, pred0, filter, k, f, lambda=NULL, r_k){
  sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], v3=data[,var3,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c("v1","v2","v3","y","pred0","filt")
  sum1 <- sqldf("SELECT v1, v2, v3, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 WHERE filt=1 GROUP BY v1, v2, v3")
  tmp1 <- sqldf("SELECT b.v1, b.v2, b.v3, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub1 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2 AND a.v3=b.v3")
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)]<-0
  tmp1$cnt1 <- tmp1$cnt
  tmp1$cnt1[filter] <- tmp1$cnt[filter] - 1
  tmp1$sumy1 <- tmp1$sumy
  tmp1$sumy1[filter] <- tmp1$sumy[filter] - tmp1$y[filter]
  tmp1$avgp <- with(tmp1, sumy1/cnt1)
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt1 - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgp + tmp1$beta*tmp1$pred0
  tmp1$avgp[is.na(tmp1$avgp)] <- tmp1$pred0[is.na(tmp1$avgp)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  tmp1$adj_avg[filter]<-tmp1$adj_avg[filter]*(1+(runif(sum(filter))-0.5)*r_k)
  return(tmp1$adj_avg)
}

cat3WayAvgCV <- function(data, var1, var2, var3, y, pred0, filter, k, f, lambda=NULL, r_k, cv=NULL){
  # It is probably best to sort your dataset first by filter and then by ID (or index)
  ind <- unlist(cv, use.names=FALSE)
  oof <- NULL
  if (length(cv) > 0){
    for (i in 1:length(cv)){
      sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], v3=data[,var3,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
      sub1 <- sub1[sub1$filt==TRUE,]
      sub1$filt <- NULL
      colnames(sub1) <- c("v1","v2","v3", "y","pred0")
      sub2 <- sub1[cv[[i]],]
      sub1 <- sub1[-cv[[i]],]
      sum1 <- sqldf("SELECT v1, v2, v3, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 GROUP BY v1, v2, v3")
      tmp1 <- sqldf("SELECT b.v1, b.v2, b.v3, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub2 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2 AND a.v3=b.v3")
      tmp1$cnt[is.na(tmp1$cnt)] <- 0
      tmp1$sumy[is.na(tmp1$sumy)] <- 0
      if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt - k)/f))
      tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
      tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
      tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
      tmp1$adj_avg <- tmp1$adj_avg*(1+(runif(nrow(sub2))-0.5)*r_k)
      oof <- c(oof, tmp1$adj_avg)
    }
  }
  oofInd <- data.frame(ind, oof)
  oofInd <- oofInd[order(oofInd$ind),]
  sub1 <- data.frame(v1=data[,var1, with=FALSE], v2=data[,var2,with=FALSE], v3=data[,var3,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c("v1","v2","v3","y","pred0","filt")
  sub2 <- sub1[sub1$filt==F,]
  sub1 <- sub1[sub1$filt==T,]
  sum1 <- sqldf("SELECT v1, v2, v3, SUM(y) as sumy, AVG(y) as avgY, sum(1) as cnt FROM sub1 GROUP BY v1, v2, v3")
  tmp1 <- sqldf("SELECT b.v1, b.v2, b.v3, b.y, b.pred0, a.sumy, a.avgY, a.cnt FROM sub2 b LEFT JOIN sum1 a ON a.v1=b.v1 AND a.v2=b.v2 AND a.v3=b.v3")
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)] <- 0
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(1+exp((tmp1$cnt - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
  tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  # Combine train and test into one vector
  return(c(oofInd$oof, tmp1$adj_avg))
}

gold_features <- function(df,number){
  list_out<-list()
  for(i in 1:number){
    list_out[[length(list_out)+1]]<-list(df[i,1],df[i,2])
  }
  list_out
}

# Combining 2 output in a parallel loop. http://stackoverflow.com/questions/19791609/saving-multiple-outputs-of-foreach-dopar-loop
comb <- function(x, ...) {
  lapply(seq_along(x),
         function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}


#######################
## Load data
#######################
t1 <- data.table(read_csv("./train.csv"))
s1 <- data.table(read_csv("./test.csv"))


s1 <- s1[,target:=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(pred0=mean(t1$target), dummy="A", filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)


# v91 and v107 are the same -- just different labels
ts1[,v107:=NULL]

excludeCols <- c("ID","target","filter","dummy","pred0")
varCols <- setdiff(colnames(ts1), excludeCols)


# Creat missingness table
charCols <- which(sapply(ts1[,-excludeCols,with=FALSE], is.character))
ts1_miss <- copy(ts1[,-excludeCols,with=FALSE])
for (col in charCols){
  set(ts1_miss, j=col, value=as.numeric(as.factor(ts1_miss[[col]])))
}
ts1_miss[!is.na(ts1_miss)] <- 0
ts1_miss[is.na(ts1_miss)] <- 1
colnames(ts1_miss) <- paste0(colnames(ts1_miss),"_NA")

# K-Means Cluster on missingness
set.seed(104)
ts1_kmeans4 <- kmeans(ts1_miss, centers=4, iter.max=50, nstart=5)
km_y_summ4 <- data.table(target=ts1$target, cluster=ts1_kmeans4$cluster, filter=ts1$filter)
km4 <- km_y_summ4[filter==0][,list(meanTarget=mean(target)), keyby=cluster] 
km_y_summ4 <- merge(km_y_summ4, km4, by="cluster")
ts1$km4 <- km_y_summ4$meanTarget

set.seed(104)
ts1_kmeans5 <- kmeans(ts1_miss, centers=5, iter.max=50, nstart=5)
km_y_summ5 <- data.table(ID=ts1$ID, target=ts1$target, cluster=ts1_kmeans5$cluster, filter=ts1$filter)
km5 <- km_y_summ5[filter==0][,list(meanTarget=mean(target)), keyby=cluster] 
km_y_summ5 <- merge(km_y_summ5, km5, by="cluster")
ts1$km5 <- km_y_summ5$meanTarget

set.seed(104)
ts1_kmeans6 <- kmeans(ts1_miss, centers=6, iter.max=50, nstart=5)
km_y_summ6 <- data.table(target=ts1$target, cluster=ts1_kmeans6$cluster, filter=ts1$filter)
km6 <- km_y_summ6[filter==0][,list(meanTarget=mean(target)), keyby=cluster] 
km_y_summ6 <- merge(km_y_summ6, km6, by="cluster")
ts1$km6 <- km_y_summ6$meanTarget

set.seed(104)
ts1_kmeans7 <- kmeans(ts1_miss, centers=7, iter.max=50, nstart=5)
km_y_summ7 <- data.table(target=ts1$target, cluster=ts1_kmeans7$cluster, filter=ts1$filter)
km7 <- km_y_summ7[filter==0][,list(meanTarget=mean(target)), keyby=cluster] 
km_y_summ7 <- merge(km_y_summ7, km7, by="cluster")
ts1$km7 <- km_y_summ7$meanTarget

numCols <- names(which(sapply(ts1[,varCols,with=FALSE], is.numeric)))
# Add row summary variables
ts1$rowMax <- apply(ts1[, numCols, with=FALSE], 1, max) 
ts1$rowMin <- apply(ts1[, numCols, with=FALSE], 1, min) 
ts1$rowMean <- apply(ts1[, numCols, with=FALSE], 1, mean)
ts1$rowMed <- apply(ts1[, numCols, with=FALSE], 1, median)
ts1$rowSD <- apply(ts1[, numCols, with=FALSE], 1, sd)
# # Create data.table with NA = -1 
# # Bind with imputed data frame
# ts1_nafill <- copy(ts1[,numCols,with=FALSE])
# ts1_nafill[is.na(ts1_nafill)] <- -1
# colnames(ts1_nafill) <- paste0(colnames(ts1_nafill),"_NAfill")

# excludeCols <- c(excludeCols, "v22")

# Count NAs by row
ts1$cntNA <- rowSums(is.na(ts1[, varCols, with=FALSE]))
ts1$cntZero <- rowSums(ts1[, varCols, with=FALSE] == 0, na.rm=TRUE)
# Give blank factor levels a name
charCols <- colnames(ts1)[sapply(ts1, is.character)]

for (i in 1:length(charCols)){
  set(ts1, i=which(is.na(ts1[[charCols[i]]])), j=charCols[i], value="NULL")
  # ts1[,charCols[i],with=FALSE]ts1[,charCols[i],with=FALSE]=="" <- "NULL"
}

#Convert character columns to factor
ts1 <- ts1[,(charCols):=lapply(.SD, as.factor),.SDcols=charCols]

#Convert integer to numeric - some functions give errors
#These variables may be ordinal
ts1$v38 <- as.factor(make.names(ts1$v38))
ts1$v62 <- as.factor(make.names(ts1$v62))
ts1$v72 <- as.factor(make.names(ts1$v72))
ts1$v129 <- as.factor(make.names(ts1$v129))


ts1[is.na(ts1)] <- -1
ts2 <- copy(ts1)

# ##################
# ## IMPUTATION
# ##################
# library(doParallel)
# # Using all cores can slow down the computer
# # significantly, I therefore try to leave one
# # core alone in order to be able to do something 
# # else during the time the code runs
# cores_2_use <- detectCores() - 2
# 
# imputeSub <- data.frame(ts1[, -excludeCols, with=FALSE])
# 
# cl <- makeCluster(cores_2_use)
# clusterSetRNGStream(cl, 9956)
# clusterExport(cl, "imputeSub")
# clusterEvalQ(cl, library(mice))
# imp_pars <- 
#   parLapply(cl = cl, X = 1:cores_2_use, fun = function(no){
#     mice(imputeSub, m = 1, maxit=1, printFlag = TRUE)
#   })
# stopCluster(cl)
# 
# imp_merged <- imp_pars[[1]]
# for (n in 2:length(imp_pars)){
#   imp_merged <- 
#     ibind(imp_merged,
#           imp_pars[[n]])
# }
# 
# save(imp_merged, file="./data_trans/mice_v14.rda")
# ts1_complete <- cbind(ts1[,excludeCols,with=FALSE], complete(imp_merged))
# write_csv(ts1_complete, "./data_trans/ts1_mice_v13.csv")

# Bind imputed data with missingness table
# ts2 <- ts1_complete

# Get rid of zero variance variables if there are any
pp <- preProcess(ts2[filter==0, -excludeCols, with=FALSE], method="zv")
ts2 <- predict(pp, ts2)

#####################
## Factor 2-way counts
#####################
factCols <- colnames(ts2[,-excludeCols,with=FALSE])[sapply(ts2[,-excludeCols,with=FALSE], is.factor)]
pairs <- combn(factCols, 2, simplify=FALSE)

cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(123)
out <- foreach(i=1:length(pairs), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
  name <- paste0(pairs[[i]][1], "_", pairs[[i]][2], "_cnt2") 
  tmp <- my.f2cnt(ts2, pairs[[i]][1], pairs[[i]][2])
  if (var(tmp[ts2$filter==0]) != 0) # exclude columns with no variance in the training set
    list(tmp, name)
               }

pairCnts <- as.data.frame(out[[1]])
colnames(pairCnts) <- unlist(out[[2]])

# 2-way averages
set.seed(123)
out <- foreach(i=1:length(pairs), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
  name <- paste0(pairs[[i]][1],"_",pairs[[i]][2], "_targetMean2way")
  tmp <- cat2WayAvgCV(data = ts2, var1 = pairs[[i]][1], var2 = pairs[[i]][2], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 20, f = 10, r_k = 0.04, cv=cvFoldsList)
  list(tmp, name)
               }

pairMeans <- as.data.frame(out[[1]])
colnames(pairMeans) <- unlist(out[[2]])

################
## Add 3-way counts
################
triplets <- combn(c("v22","v24","v47","v56", "v66","v72", "v79","v113","v125","v129"), 3, simplify=FALSE)
set.seed(123)
out <- foreach(i=1:length(triplets), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
  name <- paste0(triplets[[i]][1], "_", triplets[[i]][2],"_",triplets[[i]][3], "_cnt3") 
  tmp <- my.f3cnt(ts2, triplets[[i]][1], triplets[[i]][2], triplets[[i]][3])
  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
    list(tmp, name)
               }

tripCnts <- as.data.frame(out[[1]])
colnames(tripCnts) <- unlist(out[[2]])
# 3-way averages
set.seed(123)
out <- foreach(i=1:length(triplets), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
  name <- paste0(triplets[[i]][1],"_",triplets[[i]][2], "_", triplets[[i]][3], "_targetMean3way")
  tmp <- cat3WayAvgCV(data = ts2, var1 = triplets[[i]][1], var2 = triplets[[i]][2], var3 = triplets[[i]][3], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 30, f = 20, r_k = 0.04, cv=cvFoldsList)
  list(tmp, name)
               }

tripMeans <- as.data.frame(out[[1]])
colnames(tripMeans) <- unlist(out[[2]])

# Stop parallel cluster
stopCluster(cl)
# Combine results
ts2 <- cbind(ts2, pairCnts, pairMeans, tripCnts, tripMeans)

## Log Feature ratios from Telstra, may be useful for ratios here

# for (i in 1:length(pairs)){
#   name <- paste0(pairs[[i]][1], "_", pairs[[i]][2],"_ratio") 
#   tmp <- as.data.frame(featCast[,pairs[[i]][1], with=FALSE] / featCast[,pairs[[i]][2], with=FALSE])
#   tmp <- do.call(data.frame,lapply(tmp, function(x) replace(x, is.infinite(x), 99999)))
#   tmp <- replace(tmp, is.na(tmp), -1)
#   ts2[,name] <- tmp
# }

#####################
# 3 way interaction indicator
#####################
# triplets <- combn(charCols, 3, simplify=FALSE)
# for (i in 1:length(triplets)){
#   name <- paste0(triplets[[i]][1], "_", triplets[[i]][2], "_", triplets[[i]][3], "_int") 
#   tmp <- int3WayBool(featCast, triplets[[i]][1], triplets[[i]][2], triplets[[i]][3])
#   if (sum(tmp[ts2$filter==0]) == 0) next else # exclude columns with no variance in the training set
#     ts2[,name] <- tmp
# }

############
## PAIRWISE CORRELATIONS -- code & idea from Tian Zhou - teammate in Homesite competition
############
# Remove features with correlations equal to 1
numCols <- colnames(ts2[,-excludeCols,with=FALSE])[sapply(ts2[,-excludeCols,with=FALSE], is.numeric)]
featCor <- cor(ts2[,numCols,with=FALSE])
hc <- findCorrelation(featCor, cutoff=0.9999, names=TRUE)  
hc <- sort(hc)
if (length(hc)>0)
  ts2 <- ts2[,-hc,with=FALSE]

featCorDF <- abs(featCor[!rownames(featCor) %in% hc, !colnames(featCor) %in% hc])
featCorDF[upper.tri(featCorDF, diag=TRUE)] <- NA
featCorDF <- melt(featCorDF, varnames = c('V1','V2'), na.rm=TRUE)
featCorDF <- featCorDF[order(featCorDF$value, decreasing=TRUE),]

feat_gold30 <- gold_features(featCorDF, 100)

for (i in 1:length(feat_gold30)) {
  name <- paste0(feat_gold30[[i]][[1]],"_",feat_gold30[[i]][[2]],"_cor")
  ts2[,name] <- ts2[,as.character(feat_gold30[[i]][[1]]), with=FALSE] - ts2[,as.character(feat_gold30[[i]][[2]]), with=FALSE]
}


######################################################


############
## Helper columns
############
# Scale variables so a few don't overpower the helper columns
pp <- preProcess(ts2[filter==0,-excludeCols,with=FALSE], method=c("zv","center","scale","medianImpute"))
ts2_pp <- predict(pp, ts2)

summ <- as.data.frame(ts2_pp[ts2_pp$filter==0, colnames(ts2_pp) %in% c("target",numCols),with=FALSE] %>% group_by(target) %>%
                        summarise_each(funs(mean)))
# Find means and sd's for columns
mn1 <- sapply(summ[,2:ncol(summ)], mean)
sd1 <- sapply(summ[,2:ncol(summ)], sd)
# Find upper and lower thresholds
hi <- mn1+2*sd1
lo <- mn1-2*sd1

helpCols <- list()
for (i in 0:1){
  tmpHi <- (summ[summ$target==i,2:ncol(summ)] - mn1)/sd1
  hiNames <- colnames(tmpHi[,order(tmpHi)][,1:30])
  loNames <- colnames(tmpHi[,order(tmpHi,decreasing = TRUE)][1:30])
  
  helpCols[[i+1]] <- c(hiNames, loNames)
  
}
names(helpCols) <- paste0("X", seq_along(helpCols)-1)

for (i in 0:1){
  ts2_pp[[ncol(ts2_pp)+1]] <- rowSums(ts2_pp[,helpCols[[i+1]], with=FALSE])
  colnames(ts2_pp)[ncol(ts2_pp)] <- paste0("X", i, "_helper")
}

##################
## Create summary variables for high-dimensional factors
##################
factorCols <- colnames(ts2_pp)[sapply(ts2_pp, is.factor)]
highCardFacts <- colnames(ts2_pp[,factorCols,with=FALSE])[sapply(ts2_pp[,factorCols,with=FALSE], function(x) length(unique(x))>200)]

for(ii in highCardFacts) {
  print(ii)
  x <- data.frame(x1=ts2_pp[, ii,with=FALSE])
  x[,ii] <- as.numeric(x[,ii])
  ts2_pp[, paste(ii, "_num", sep="")] <- x
}


for(ii in highCardFacts) {
  print(ii)
  x <- data.frame(x1=ts2_pp[, ii,with=FALSE])
  colnames(x) <- "x1"
  x$x1 <- as.numeric(x$x1)
  sum1 <- sqldf("select x1, sum(1) as cnt
                from x  group by 1 ")
  tmp <- sqldf("select cnt from x a left join sum1 b on a.x1=b.x1")
  ts2_pp[, paste(ii, "_cnt", sep="")] <- tmp$cnt
}

# Replace high cardinality factors with target mean
for(ii in highCardFacts) {
  name <- paste0(highCardFacts, "_targetMean")
  ts2_pp[,name] <- cat2WayAvg(data = ts2_pp, var1 = highCardFacts, var2 = "dummy", y = "target",pred0 = "pred0",filter = ts2_pp$filter==0, k = 20, f = 10, r_k = 0.05)
  }


ts2_pp <- ts2_pp[,!colnames(ts2_pp) %in% highCardFacts,with=FALSE]

##################
## Create dummy variables for low-dimensional factors
##################

dummy <- dummyVars( ~. -1, data = ts2_pp[,-c("dummy","pred0"),with=FALSE])
ts2_dum <- data.frame(predict(dummy, ts2_pp))


# # Bind miputed data to data with -1 as NA
# ts2_dum <- cbind(ts2_dum, ts1_nafill)
# # Remove highly correlated columns (some nafill columns might be the same)
# numCols <- colnames(ts2_dum[,!colnames(ts2_dum) %in% excludeCols])[sapply(ts2_dum[,!colnames(ts2_dum) %in% excludeCols], is.numeric)]
# featCor <- cor(ts2_dum[,numCols])
# hc <- findCorrelation(featCor, cutoff=0.999999, names=TRUE)  
# hc <- sort(hc)
# ts2_dum <- ts2_dum[,!colnames(ts2_dum) %in% hc]

###################
## Write CSV file
###################
write.csv(as.data.frame(helpCols), "./data_trans/helpCols_v14.csv", row.names=FALSE)
save(helpCols, file="./data_trans/helpCols_v14.rda")

ts2_dum <- ts2_dum[order(ts2_dum$filter, ts2_dum$ID),]
write_csv(ts2_dum, "./data_trans/ts2Trans_v14.csv")




