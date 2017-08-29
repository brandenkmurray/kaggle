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
library(WGCNA)
library(VGAM)
# library(Boruta)
setwd("/media/branden/SSHD1/kaggle/bnp")
# load("./data_trans/cvFoldsList.rda")
threads <- detectCores() - 2
##################
## FUNCTIONS
#################
source("./data_trans/utils.R")

#################
## Dmitry's holdout
#################
folds <- read.csv("./Dmitry/holdout.csv")


##################
## Dmitry's feature interactions
##################
comb2List <- list(c("v50","v6"),c("v21","v5"),c("v10","v12"),c("v50","v78"),
                  c("v115","v52"),c("v21","v24"),c("v50","v66"),c("v30","v40"),
                  c("v129","v24"),c("v3","v50"),c("v39","v66"),c("v34","v56"),
                  c("v113","v12"),c("v114","v22"),c("v120","v50"),c("v50","v90"),
                  c("v12","v52"),c("v34","v91"),c("v12","v58"),c("v14","v22"),
                  c("v1","v45"),c("v40","v66"),c("v125","v34"),c("v2","v21"),
                  c("v66","v74"),c("v31","v56"),c("v114","v12"),c("v24","v30"),
                  c("v125","v22"),c("v113","v22"),c("v125","v52"),c("v39","v50"),
                  c("v12","v42"),c("v10","v47"),c("v112","v34"),c("v21","v72"),
                  c("v41","v49"),c("v10","v66"),c("v50","v98"),c("v10","v114"),
                  c("v21","v74"),c("v113","v21"),c("v12","v66"),c("v24","v45"),
                  c("v21","v45"),c("v50","v52"),c("v131","v50"),c("v12","v88"),
                  c("v1","v40"),c("v34","v40"),c("v24","v50"),c("v22","v56"),
                  c("v113","v80"),c("v122","v50"),c("v14","v38"),c("v10","v91"),
                  c("v49","v56"),c("v10","v56"),c("v50","v68"),c("v16","v34"),
                  c("v31","v66"),c("v106","v56"),c("v112","v14"),c("v34","v52"),
                  c("v50","v69"),c("v113","v56"),c("v12","v47"),c("v10","v24"),
                  c("v50","v80"),c("v125","v47"),c("v113","v66"),c("v21","v69"),
                  c("v113","v50"),c("v38","v56"),c("v115","v34"),c("v14","v39"),
                  c("v14","v72"),c("v14","v50"),c("v113","v131"),c("v21","v56"),
                  c("v114","v47"),c("v125","v14"),c("v113","v24"),c("v50","v91"),
                  c("v12","v99"),c("v125","v24"),c("v58","v85"),c("v1","v50"),
                  c("v56","v85"),c("v21","v47"),c("v12","v71"),c("v125","v18"),
                  c("v21","v6"),c("v50","v72"),c("v131","v34"),c("v40","v52"),
                  c("v21","v58"),c("v10","v21"),c("v127","v56"),c("v66","v71"),
                  c("v34","v50"),c("v131","v27"),c("v30","v66"),c("v45","v56"),
                  c("v100","v114"),c("v45","v66"),c("v56","v88"),c("v14","v78"),
                  c("v21","v85"),c("v112","v56"),c("v100","v80"),c("v30","v34"),
                  c("v114","v40"),c("v50","v99"),c("v44","v56"),c("v10","v125"),
                  c("v12","v18"),c("v14","v21"),c("v113","v14"),c("v114","v52"),
                  c("v47","v56"),c("v56","v80"),c("v24","v40"),c("v100","v14"),
                  c("v45","v50"),c("v113","v47"),c("v22","v40"),c("v113","v34"),
                  c("v1","v56"),c("v30","v91"),c("v112","v40"),c("v112","v21"),
                  c("v24","v72"),c("v22","v52"),c("v31","v47"),c("v30","v56"),
                  c("v10","v62"),c("v10","v40"),c("v38","v66"),c("v56","v66"),
                  c("v21","v52"),c("v112","v114"),c("v31","v50"),c("v113","v125"),
                  c("v100","v66"),c("v14","v66"),c("v34","v47"),c("v38","v50"),
                  c("v1","v14"),c("v24","v66"),c("v40","v47"),c("v24","v56"),
                  c("v14","v24"),c("v50","v7"),c("v100","v27"),c("v114","v14"),
                  c("v10","v50"),c("v114","v50"),c("v40","v74"),c("v12","v30"),
                  c("v115","v21"),c("v112","v113"),c("v24","v31"),c("v125","v56"),
                  c("v56","v99"),c("v52","v6"),c("v12","v56"),c("v100","v56"),
                  c("v14","v47"),c("v114","v71"),c("v113","v45"),c("v10","v14"),
                  c("v56","v68"),c("v21","v22"),c("v50","v85"),c("v40","v50"),
                  c("v12","v40"),c("v129","v66"),c("v19","v50"),c("v47","v66"),
                  c("v113","v114"),c("v34","v51"),c("v50","v6"),c("v56","v72"),
                  c("v129","v50"),c("v125","v21"),c("v50","v56"),c("v22","v34"),
                  c("v114","v58"),c("v12","v91"),c("v21","v40"),c("v24","v68"),
                  c("v14","v80"),c("v39","v6"),c("v131","v6"),c("v12","v39"),
                  c("v12","v21"),c("v12","v125"),c("v12","v14"),c("v56","v87"),
                  c("v45","v47"),c("v24","v6"),c("v113","v72"),c("v12","v50"),
                  c("v14","v40"),c("v100","v40"),c("v47","v50"),c("v56","v58"),
                  c("v24","v34"),c("v115","v50"),c("v24","v62"),c("v39","v56"),
                  c("v100","v24"),c("v52","v56"),c("v18","v21"),c("v12","v24"),
                  c("v113","v40"),c("v30","v47"),c("v12","v5"),c("v50","v60"),
                  c("v14","v52"),c("v127","v98"),c("v12","v120"),c("v50","v74"),
                  c("v66","v72"),c("v112","v52"),c("v100","v50"),c("v45","v72"),
                  c("v40","v56"),c("v34","v6"),c("v125","v40"),c("v125","v50"),
                  c("v114","v34"),c("v34","v35"),c("v127","v39"),c("v114","v66"),
                  c("v24","v47"),c("v45","v90"),c("v12","v38"),c("v50","v97"),
                  c("v21","v5"),c("v30","v50"),c("v112","v12"),c("v14","v34"),
                  c("v120","v45"),c("v38","v78"),c("v114","v125"),c("v114","v56"),
                  c("v127","v21"),c("v114","v21"),c("v22","v50"),c("v12","v34"),
                  c("v34","v66"),c("v14","v56"),c("v12","v22"),c("v40","v99"),
                  c("v11","v50"),c("v12","v85"),c("v29","v77"),c("v24","v52"),
                  c("v12","v78"),c("v28","v50"),c("v112","v47"),c("v114","v24"),
                  c("v80","v99"),c("v129","v56"),c("v127","v19"),c("v14","v99"),
                  c("v21","v34"),c("v10","v52"),c("v10","v113"),c("v10","v34"),
                  c("v112","v50"),c("v100","v47"),c("v21","v66"),c("v1","v120"),
                  c("v115","v12"))



comb3List <- list(c("v12","v24","v56"),c("v113","v14","v40"),c("v113","v14","v66"),c("v38","v50","v78"),
                  c("v34","v40","v66"),c("v12","v14","v50"),c("v56","v66","v72"),c("v21","v24","v50"),c("v113","v31","v56"),
                  c("v113","v14","v34"),c("v21","v47","v50"),c("v12","v47","v56"),c("v40","v50","v66"),
                  c("v38","v47","v50"),c("v129","v50","v56"),c("v24","v40","v66"),c("v113","v31","v50"),
                  c("v10","v12","v14"),c("v24","v50","v66"),c("v31","v50","v66"),c("v10","v34","v66"),
                  c("v12","v24","v47"),c("v12","v38","v50"),c("v12","v50","v66"),c("v50","v60","v66"),
                  c("v14","v21","v40"),c("v10","v21","v66"),c("v10","v40","v66"),c("v31","v47","v66"),
                  c("v21","v50","v66"),c("v113","v40","v50"),c("v114","v14","v66"),c("v10","v14","v50"),
                  c("v14","v34","v50"),c("v34","v56","v66"),c("v34","v47","v50"),c("v40","v50","v74"),
                  c("v114","v40","v66"),c("v125","v34","v40"),c("v30","v40","v47"),c("v131","v50","v66"),
                  c("v114","v47","v66"),c("v10","v14","v40"),c("v45","v50","v72"),c("v10","v50","v66"),
                  c("v31","v56","v66"),c("v12","v34","v66"),c("v113","v114","v24"),c("v10","v12","v125"),
                  c("v12","v40","v66"),c("v34","v51","v66"),c("v10","v113","v12"),c("v100","v40","v66"),
                  c("v50","v56","v80"),c("v50","v66","v97"),c("v10","v47","v56"),c("v12","v14","v56"),
                  c("v129","v31","v50"),c("v114","v56","v66"),c("v50","v66","v72"),c("v114","v50","v56"),
                  c("v113","v129","v66"),c("v10","v114","v50"),c("v28","v50","v66"),c("v12","v50","v56"),
                  c("v10","v50","v56"),c("v10","v40","v50"),c("v113","v47","v50"),c("v10","v114","v66"),
                  c("v114","v14","v47"),c("v14","v47","v50"),c("v40","v66","v74"),c("v14","v38","v66"),
                  c("v10","v21","v50"),c("v50","v66","v93"),c("v14","v21","v50"),c("v21","v40","v56"),
                  c("v50","v56","v68"),c("v129","v56","v66"),c("v12","v14","v66"),c("v10","v12","v66"),
                  c("v10","v114","v12"),c("v34","v50","v66"),c("v100","v113","v47"),c("v14","v38","v50"),
                  c("v113","v50","v72"),c("v10","v14","v56"),c("v24","v34","v56"),c("v14","v47","v66"),
                  c("v12","v24","v66"),c("v10","v12","v50"),c("v14","v56","v66"),c("v113","v24","v66"),
                  c("v21","v50","v74"),c("v129","v14","v66"),c("v113","v114","v66"),c("v12","v31","v66"),
                  c("v30","v47","v66"),c("v30","v40","v66"),c("v47","v56","v66"),c("v12","v47","v66"),
                  c("v45","v56","v66"),c("v21","v24","v66"),c("v12","v14","v40"),c("v113","v56","v66"),
                  c("v24","v50","v72"),c("v10","v12","v21"),c("v24","v56","v66"),c("v24","v50","v56"),
                  c("v100","v47","v50"),c("v50","v66","v74"),c("v113","v12","v66"),c("v14","v34","v40"),
                  c("v14","v50","v99"),c("v12","v21","v66"),c("v10","v31","v66"),c("v47","v50","v56"),
                  c("v14","v50","v56"),c("v40","v66","v7"),c("v30","v34","v47"),c("v129","v40","v50"),
                  c("v50","v66","v7"),c("v38","v50","v66"),c("v10","v113","v14"),c("v14","v34","v66"),
                  c("v34","v50","v56"),c("v114","v12","v40"),c("v114","v47","v50"),c("v100","v47","v66"),
                  c("v24","v47","v56"),c("v34","v47","v66"),c("v10","v113","v24"),c("v30","v50","v66"),
                  c("v114","v14","v50"),c("v114","v24","v47"),c("v113","v24","v31"),c("v113","v34","v40"),
                  c("v50","v62","v66"),c("v114","v12","v56"),c("v24","v31","v56"),c("v40","v66","v71"),
                  c("v31","v47","v50"),c("v34","v66","v71"),c("v129","v24","v66"),c("v14","v24","v66"),
                  c("v10","v114","v47"),c("v10","v56","v66"),c("v113","v14","v24"),c("v14","v40","v47"),
                  c("v10","v21","v56"),c("v21","v40","v66"),c("v125","v50","v66"),c("v12","v21","v50"),
                  c("v10","v14","v21"),c("v40","v66","v72"),c("v40","v50","v56"),c("v24","v40","v47"),
                  c("v38","v47","v66"),c("v24","v31","v50"),c("v14","v47","v56"),c("v10","v113","v66"),
                  c("v129","v24","v56"),c("v21","v56","v85"),c("v129","v40","v66"),c("v21","v47","v56"),
                  c("v31","v40","v50"),c("v10","v47","v66"),c("v113","v40","v47"),c("v10","v47","v50"),
                  c("v113","v47","v66"),c("v10","v12","v40"),c("v14","v40","v66"),c("v113","v114","v50"),
                  c("v113","v40","v56"),c("v10","v50","v52"),c("v30","v34","v66"),c("v12","v21","v56"),
                  c("v34","v47","v56"),c("v114","v24","v50"),c("v24","v45","v56"),c("v50","v56","v88"),
                  c("v21","v56","v66"),c("v40","v47","v66"),c("v10","v12","v24"),c("v21","v34","v66"),
                  c("v10","v12","v47"),c("v24","v30","v47"),c("v10","v12","v56"),c("v40","v47","v50"),
                  c("v114","v47","v56"),c("v34","v40","v50"),c("v113","v24","v56"),c("v50","v66","v71"),
                  c("v114","v34","v47"),c("v114","v40","v50"),c("v114","v34","v66"),c("v14","v31","v50"),
                  c("v100","v24","v47"),c("v21","v50","v56"),c("v24","v31","v66"),c("v114","v21","v50"),
                  c("v113","v12","v14"),c("v14","v40","v50"),c("v24","v34","v50"),c("v21","v24","v47"),
                  c("v14","v34","v47"),c("v10","v24","v66"),c("v114","v31","v50"),c("v3","v50","v66"),
                  c("v10","v12","v34"),c("v113","v24","v34"),c("v113","v40","v66"),c("v24","v40","v50"),
                  c("v39","v50","v66"),c("v24","v47","v66"),c("v113","v14","v47"),c("v113","v14","v50"),
                  c("v21","v40","v50"),c("v24","v47","v50"),c("v100","v50","v66"),c("v10","v24","v56"),
                  c("v129","v50","v66"),c("v12","v56","v66"),c("v113","v12","v24"),c("v34","v40","v47"),
                  c("v113","v114","v47"),c("v113","v24","v50"),c("v113","v24","v47"),c("v24","v40","v56"),
                  c("v1","v50","v56"),c("v24","v34","v66"),c("v24","v34","v47"),c("v114","v66","v71"),
                  c("v31","v45","v56"),c("v114","v24","v66"),c("v113","v31","v66"),c("v114","v50","v66"),
                  c("v10","v34","v50"),c("v21","v40","v47"),c("v40","v56","v66"),c("v14","v24","v47"),
                  c("v113","v34","v50"),c("v113","v34","v66"),c("v113","v50","v66"),c("v10","v24","v47"),
                  c("v50","v56","v66"),c("v40","v47","v56"),c("v31","v50","v56"),c("v47","v50","v66"),
                  c("v24","v31","v47"),c("v24","v38","v50"),c("v113","v24","v40"),c("v113","v50","v56"),
                  c("v113","v34","v47"),c("v14","v24","v50"),c("v12","v14","v24"),c("v14","v50","v66"),
                  c("v10","v14","v66"),c("v31","v40","v66"))


#######################
## Load data
#######################
t1 <- fread("./train.csv")
s1 <- fread("./test.csv")


s1 <- s1[,target:=-1]
# Combine into 1 data frame
l <- list(t1, s1)
ts1 <- data.table(do.call(smartbind,l))
ts1 <- cbind(pred0=mean(t1$target), dummy="A", filter=c(rep(0, nrow(t1)), rep(2, nrow(s1))), ts1)


# v91 and v107 are the same -- just different labels
ts1[,v107:=NULL]
# v10 -- round v10 -- convert to categorical later
ts1[,v10:=round(v10/0.0218818357511,0)]


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
ts1_kmeans7 <- kmeans(ts1_miss, centers=7, iter.max=50, nstart=5)
km_y_summ7 <- data.table(target=ts1$target, cluster=ts1_kmeans7$cluster, filter=ts1$filter)
km7 <- km_y_summ7[filter==0][,list(meanTarget=mean(target)), keyby=cluster] 
km_y_summ7 <- merge(km_y_summ7, km7, by="cluster")
ts1$km7 <- as.factor(make.names(km_y_summ7$cluster))

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
# round numeric values
ts1[,(numCols) := round(.SD,4), .SDcols=numCols]
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
ts1$v10 <- as.factor(make.names(ts1$v10))
ts1$v38 <- as.factor(make.names(ts1$v38))
ts1$v62 <- as.factor(make.names(ts1$v62))
ts1$v72 <- as.factor(make.names(ts1$v72))
ts1$v129 <- as.factor(make.names(ts1$v129))

# #Box-Cox Transform numerics
# bc <- function(x) {
#   bc <- BoxCoxTrans(x+1e-5, na.rm=TRUE)
#   return(predict(bc, x+1e-5))
# }
# 
# #Notes:
# # median impute insteaf of -1
# # dummy vars km7?
# numCols <- colnames(ts1[,-excludeCols,with=FALSE])[sapply(ts1[,-excludeCols,with=FALSE], is.numeric)]
# ts1[,(numCols):=lapply(.SD, bc), .SDcols=numCols]
# 
# pp <- preProcess(ts1[,numCols,with=F], method=c("medianImpute","center","scale"))
# ts1 <- predict(pp, ts1)

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
## Numeric interactions
#####################
pairs <- combn(c("v35","v21","v12","v50","v14","v40","v114","v34"), 2, simplify=FALSE)
# v35 & v21 turned out ot be the only important interaction
cl <- makeCluster(threads, type="FORK")
registerDoParallel(cl)
set.seed(119)
out <- foreach(i=1:length(pairs), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("data.table")) %dorng% {
                 name <- paste0(pairs[[i]][1], "_", pairs[[i]][2], "_int2") 
                 tmp <- ts2[,pairs[[i]][1], with=FALSE] / (ts2[,pairs[[i]][2], with=FALSE] + 1e-05)
                 if (var(tmp[ts2$filter==0]) != 0) # exclude columns with no variance in the training set
                   list(tmp, name)
               }
stopCluster(cl)
pairInts <- as.data.frame(out[[1]])
colnames(pairInts) <- unlist(out[[2]])

ts2 <- cbind(ts2, pairInts)
rm(pairInts); gc()

#####################
## Factor 2-way counts
#####################
factCols <- colnames(ts2[,-excludeCols,with=FALSE])[sapply(ts2[,-excludeCols,with=FALSE], is.factor)]
pairs <- combn(factCols, 2, simplify=FALSE)

cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(120)
out <- foreach(i=1:length(pairs), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
                 name <- paste0(pairs[[i]][1], "_", pairs[[i]][2], "_cnt2") 
                 tmp <- my.f2cnt(ts2, pairs[[i]][1], pairs[[i]][2])
                 if (var(tmp[ts2$filter==0]) != 0) # exclude columns with no variance in the training set
                   list(tmp, name)
               }
stopCluster(cl)
pairCnts <- as.data.frame(out[[1]])
colnames(pairCnts) <- unlist(out[[2]])

ts2 <- cbind(ts2, pairCnts)
rm(pairCnts); gc()

# 2-way averages
cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(121)
out <- foreach(i=1:length(pairs), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
                 name <- paste0(paste0(pairs[[i]],collapse="_"), "_targetMean2way")
                 tmp <- catNWayAvgCV(data = ts2, pairs[[i]], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 0, f = 0, r_k = 0.00, cv=cvFoldsList)
                 tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
                 list(tmp, name)
               }
stopCluster(cl)
pairMeans <- as.data.frame(out[[1]])
colnames(pairMeans) <- unlist(out[[2]])


ts2 <- cbind(ts2, pairMeans)
rm(pairMeans); gc()
################
## Add 3-way counts
################
triplets <- combn(c("v3","v10","v22","v24","v30","v38","v47", "v52", "v56", "v62","v66","v72","v74", "v75","v79", "v91","v110","v112","v113","v125","v129"), 3, simplify=FALSE)
# triplets <- combn(c("v3","v22", "v52", "v56","v66","v72","v79", "v110","v113"), 3, simplify=FALSE)
# triplets <- combn(factCols, 3, simplify=FALSE)
# triplets2 <- combn(setdiff(factCols, c("v3","v22","v24","v30","v38","v47", "v52", "v56", "v62","v66","v72","v74", "v75","v79", "v91","v110","v112","v113","v125","v129")), 3, simplify=FALSE)
# set.seed(900)
# triplets <- combn(sample(factCols, size = 12, replace = FALSE), 3, simplify=FALSE)
cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(122)
out <- foreach(i=1:length(triplets), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
                 name <- paste0(triplets[[i]][1], "_", triplets[[i]][2],"_",triplets[[i]][3], "_cnt3") 
                 tmp <- my.f3cnt(ts2, triplets[[i]][1], triplets[[i]][2], triplets[[i]][3])
                 if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
                   list(tmp, name)
               }
stopCluster(cl)
tripCnts <- as.data.frame(out[[1]])
colnames(tripCnts) <- unlist(out[[2]])

ts2 <- cbind(ts2, tripCnts)
rm(tripCnts); gc()


# 3-way averages
triplets <- combn(c("v3","v10","v22","v24","v30","v38","v47", "v52", "v56", "v62","v66","v72","v74", "v75","v79", "v91","v110","v112","v113","v125","v129"), 3, simplify=FALSE)
# add v22
comb2List_v22 <- lapply(comb2List, function(x) c(x,"v22"))
triplets <- c(triplets, comb2List_v22)
# sort vectors and remove duplicates
triplets <- unique(lapply(triplets, function(x) sort(x)))

# triplets <- combn(factCols, 3, simplify=FALSE)
cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(123)
out <- foreach(i=1:length(triplets), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
                 name <- paste0(paste0(triplets[[i]],collapse="_"), "_targetMean3way")
                 tmp <- catNWayAvgCV(data = ts2, triplets[[i]], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 0, f = 0, r_k = 0.00, cv=cvFoldsList)
                 tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
                 list(tmp, name)
               }
stopCluster(cl)
tripMeans <- as.data.frame(out[[1]])
colnames(tripMeans) <- unlist(out[[2]])

ts2 <- cbind(ts2, tripMeans)
rm(tripMeans); gc()

################
## Add 4-way counts
################
# quads <- combn(c("v3","v22","v24","v47","v52","v56", "v66","v72", "v74", "v79","v113","v125","v129"), 4, simplify=FALSE)
# # set.seed(901)
# # quads2 <- combn(sample(setdiff(factCols, c("v3","v22","v24","v47","v52","v56", "v66","v72", "v74", "v79","v113","v125","v129")),10,replace=FALSE), 4, simplify=FALSE)
# # set.seed(901)
# # # quads <- combn(sample(factCols, 10, replace=FALSE), 4, simplify=FALSE)
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(125)
# out <- foreach(i=1:length(quads), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
#                  name <- paste0(quads[[i]][1], "_", quads[[i]][2],"_",quads[[i]][3], "_",quads[[i]][4],"_cnt4") 
#                  tmp <- my.f4cnt(ts2, quads[[i]][1], quads[[i]][2], quads[[i]][3], quads[[i]][4])
#                  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
#                    list(tmp, name)
#                }
# stopCluster(cl)
# quadCnts <- as.data.frame(out[[1]])
# colnames(quadCnts) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, quadCnts)
# rm(quadCnts); gc()

# 4-way averages
# quads <- combn(c("v3","v22","v24","v47","v52","v56", "v66","v72", "v74", "v79","v113","v125","v129"), 4, simplify=FALSE)
# add v22
quads <- lapply(comb3List, function(x) c(x,"v22"))
# quads <- c(triplets, comb3List_v22)
# sort vectors and remove duplicates
# quads <- unique(lapply(triplets, function(x) sort(x)))
cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(127)
out <- foreach(i=1:length(quads), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
                 name <- paste0(paste0(quads[[i]],collapse="_"), "_targetMean4way")
                 tmp <- catNWayAvgCV(data = ts2, quads[[i]], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 0, f = 0, r_k = 0.00, cv=cvFoldsList)
                 tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
                 list(tmp, name)
               }
stopCluster(cl)
quadMeans <- as.data.frame(out[[1]])
colnames(quadMeans) <- unlist(out[[2]])

ts2 <- cbind(ts2, quadMeans)
rm(quadMeans); gc()


################
## Add 5-way counts
################
# quints <- combn(c("v22","v24","v47","v52","v56", "v66","v72", "v79","v113","v125","v129"), 5, simplify=FALSE)
# # set.seed(902)
# # quints2 <- combn(sample(setdiff(factCols, c("v22","v24","v47","v52","v56", "v66","v72", "v79","v113","v125","v129")),10,replace=FALSE), 5, simplify=FALSE)
# # set.seed(902)
# # quints <- combn(sample(factCols, 10, replace=FALSE), 5, simplify=FALSE)
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(129)
# out <- foreach(i=1:length(quints), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
#                  name <- paste0(quints[[i]][1], "_", quints[[i]][2],"_",quints[[i]][3], "_",quints[[i]][4], "_",quints[[i]][5],"_cnt5") 
#                  tmp <- my.f5cnt(ts2, quints[[i]][1], quints[[i]][2], quints[[i]][3], quints[[i]][4], quints[[i]][5])
#                  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
#                    list(tmp, name)
#                }
# stopCluster(cl)
# quintCnts <- as.data.frame(out[[1]])
# colnames(quintCnts) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, quintCnts)
# rm(quintCnts); gc()
# 
# # 5-way averages
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(131)
# out <- foreach(i=1:length(quints), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
#                  name <- paste0(quints[[i]][1],"_",quints[[i]][2], "_", quints[[i]][3],  "_", quints[[i]][4],  "_", quints[[i]][5],"_targetMean5way")
#                  tmp <- cat5WayAvgCV(data = ts2, var1 = quints[[i]][1], var2 = quints[[i]][2], var3 = quints[[i]][3], var4 = quints[[i]][4], var5 = quints[[i]][5], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 20, f = 10, r_k = 0.03, cv=cvFoldsList10)
#                  tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
#                  list(tmp, name)
#                }
# stopCluster(cl)
# quintMeans <- as.data.frame(out[[1]])
# colnames(quintMeans) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, quintMeans)
# rm(quintMeans); gc()


################
## Add 6-way counts
################
# #Remove v52, add v110
# sextups <- combn(c("v22","v24","v47","v56", "v66","v72", "v79","v110","v113","v125","v129"), 6, simplify=FALSE)
# # set.seed(903)
# # sextups2 <- combn(sample(setdiff(factCols, c("v22","v24","v47","v56", "v66","v72", "v79","v110","v113","v125","v129")), 10, replace=FALSE), 6, simplify=FALSE)
# # set.seed(903)
# # sextups <- combn(sample(factCols, 10, replace=FALSE), 6, simplify=FALSE)
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(133)
# out <- foreach(i=1:length(sextups), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
#                  name <- paste0(sextups[[i]][1], "_", sextups[[i]][2],"_",sextups[[i]][3], "_",sextups[[i]][4], "_",sextups[[i]][5], "_",sextups[[i]][6], "_cnt6") 
#                  tmp <- my.f6cnt(ts2, sextups[[i]][1], sextups[[i]][2], sextups[[i]][3], sextups[[i]][4], sextups[[i]][5], sextups[[i]][6])
#                  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
#                    list(tmp, name)
#                }
# stopCluster(cl)
# sextupCnts <- as.data.frame(out[[1]])
# colnames(sextupCnts) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, sextupCnts)
# rm(sextupCnts); gc()
# 
# # 6-way averages
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(135)
# out <- foreach(i=1:length(sextups), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
#                  name <- paste0(sextups[[i]][1],"_",sextups[[i]][2], "_", sextups[[i]][3],  "_", sextups[[i]][4],  "_", sextups[[i]][5], "_", sextups[[i]][6], "_targetMean6way")
#                  tmp <- cat6WayAvgCV(data = ts2, var1 = sextups[[i]][1], var2 = sextups[[i]][2], var3 = sextups[[i]][3], var4 = sextups[[i]][4], var5 = sextups[[i]][5], var6 = sextups[[i]][6], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 20, f = 10, r_k = 0.03, cv=cvFoldsList10)
#                  tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
#                  list(tmp, name)
#                }
# stopCluster(cl)
# sextupMeans <- as.data.frame(out[[1]])
# colnames(sextupMeans) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, sextupMeans)
# rm(sextupMeans); gc()

################
## Add 7-way counts
################
#Remove v52, add v110
septups <- combn(c("v22","v10","v24","v47","v52","v56", "v66","v72", "v74", "v79","v110","v113","v125","v129"), 7, simplify=FALSE)
# set.seed(903)
# septups2 <- combn(sample(setdiff(factCols, c("v22","v24","v47","v56", "v66","v72", "v79","v110","v113","v125","v129")), 10, replace=FALSE), 6, simplify=FALSE)
# set.seed(903)
# septups <- combn(sample(factCols, 10, replace=FALSE), 7, simplify=FALSE)

# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(133)
# out <- foreach(i=1:length(septups), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
#                  name <- paste0(septups[[i]][1], "_", septups[[i]][2],"_",septups[[i]][3], "_",septups[[i]][4], "_",septups[[i]][5], "_",septups[[i]][6], "_",septups[[i]][7], "_cnt7") 
#                  tmp <- my.f7cnt(ts2, septups[[i]][1], septups[[i]][2], septups[[i]][3], septups[[i]][4], septups[[i]][5], septups[[i]][6], septups[[i]][7])
#                  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
#                    list(tmp, name)
#                }
# stopCluster(cl)
# septupsCnts <- as.data.frame(out[[1]])
# colnames(septupsCnts) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, septupCnts)
# rm(septupCnts); gc()

# 7-way averages
cl <- makeCluster(threads)
registerDoParallel(cl)
set.seed(135)
out <- foreach(i=1:length(septups), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
                 name <- paste0(paste0(septups[[i]],collapse="_"), "_targetMean7way")
                 tmp <- catNWayAvgCV(data = ts2, septups[[i]], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 0, f = 0, r_k = 0.00, cv=cvFoldsList)
                 tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
                 list(tmp, name)
               }
stopCluster(cl)
septupsMeans <- as.data.frame(out[[1]])
colnames(septupsMeans) <- unlist(out[[2]])

ts2 <- cbind(ts2, septupsMeans)
rm(septupsMeans); gc()

################
## Add 9-way counts
################
# nonuplets <- combn(c("v3","v22","v24","v47","v52","v56", "v66","v72", "v74", "v79","v91","v110","v113","v125","v129"), 9, simplify=FALSE)
# # set.seed(903)
# # nonuplets2 <- combn(sample(setdiff(factCols, c("v22","v24","v47","v56", "v66","v72", "v79","v110","v113","v125","v129")), 10, replace=FALSE), 6, simplify=FALSE)
# # set.seed(903)
# # nonuplets <- combn(sample(factCols, 10, replace=FALSE), 9, simplify=FALSE)
# 
# # cl <- makeCluster(threads)
# # registerDoParallel(cl)
# # set.seed(133)
# # out <- foreach(i=1:length(nonuplets), .combine='comb', .multicombine=TRUE,
# #                .init=list(list(), list()), .packages=c("sqldf", "data.table")) %dorng% {
# #                  name <- paste0(nonuplets[[i]][1], "_", nonuplets[[i]][2],"_",nonuplets[[i]][3], "_",nonuplets[[i]][4], "_",nonuplets[[i]][5], "_",nonuplets[[i]][6], "_",nonuplets[[i]][7],nonuplets[[i]][8],nonuplets[[i]][9], "_cnt9") 
# #                  tmp <- my.f9cnt(ts2, nonuplets[[i]][1], nonuplets[[i]][2], nonuplets[[i]][3], nonuplets[[i]][4], nonuplets[[i]][5], nonuplets[[i]][6], nonuplets[[i]][7])
# #                  if (var(tmp[ts2$filter==0]) != 0)  # exclude columns with no variance in the training set
# #                    list(tmp, name)
# #                }
# # stopCluster(cl)
# # nonupletsCnts <- as.data.frame(out[[1]])
# # colnames(nonupletsCnts) <- unlist(out[[2]])
# # 
# # ts2 <- cbind(ts2, nonupletsCnts)
# # rm(septupCnts); gc()
# 
# # # 9-way averages
# cl <- makeCluster(threads)
# registerDoParallel(cl)
# set.seed(135)
# out <- foreach(i=1:length(nonuplets), .combine='comb', .multicombine=TRUE,
#                .init=list(list(), list()), .packages=c("sqldf", "data.table","VGAM")) %dorng% {
#                  name <- paste0(paste0(nonuplets[[i]],collapse="_"), "_targetMean9way")
#                  tmp <- catNWayAvgCV(data = ts2, nonuplets[[i]], y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 10, f = 10, r_k = 0.03, cv=cvFoldsList10)
#                  tmp <- logit(pmin(pmax(tmp, 1e-15), 1-1e-15))
#                  list(tmp, name)
#                }
# stopCluster(cl)
# nonupletsMeans <- as.data.frame(out[[1]])
# colnames(nonupletsMeans) <- unlist(out[[2]])
# 
# ts2 <- cbind(ts2, nonupletsMeans)
# rm(nonupletsMeans); gc()
# Combine results
# ts2 <- cbind(ts2, pairInts, pairCnts, pairMeans, tripCnts, tripMeans,  quadCnts, quadMeans, quintCnts, quintMeans, sextupCnts, sextupMeans)
# rm(pairInts, pairCnts, pairMeans, tripCnts, tripMeans,  quadCnts, quadMeans, quintCnts, quintMeans,  sextupCnts, sextupMeans)

# ts2 <- cbind(ts2, pairInts, pairCnts, pairMeans, tripCnts, tripMeans, quintCnts, quintMeans, septupsCnts, septupsMeans)
# rm(pairInts, pairCnts, pairMeans, tripCnts, tripMeans, quintCnts, quintMeans,  septupsCnts, septupsMeans)
# gc()
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
# boruta_results <- Boruta(as.factor(target) ~ ., ts2[filter==0,-c("dummy","pred0","ID","filter"),with=FALSE], holdHistory=FALSE, maxRuns=100) 
enableWGCNAThreads(threads)
featCor <- corFast(ts2[,numCols,with=FALSE], nThreads=threads)
hc <- findCorrelation(featCor, cutoff=0.995,names=TRUE)  
hc <- sort(hc)
save(featCor, file="./data_trans/featCor_v30.rda")



featCorDF <- abs(featCor[!rownames(featCor) %in% hc, !colnames(featCor) %in% hc])
featCorDF[upper.tri(featCorDF, diag=TRUE)] <- NA
featCorDF <- melt(featCorDF, varnames = c('V1','V2'), na.rm=TRUE)
featCorDF <- featCorDF[order(featCorDF$value, decreasing=TRUE),]

goldFeats <- 300
feat_gold <- gold_features(featCorDF, goldFeats)

# Do not parallelize -- too much memory for some reason
set.seed(136)
out <- foreach(i=1:length(feat_gold), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("data.table")) %dorng% {
                 name <- paste0(feat_gold[[i]][[1]],"_",feat_gold[[i]][[2]],"_cor")
                 tmp <- ts2[,as.character(feat_gold[[i]][[1]]), with=FALSE] - ts2[,as.character(feat_gold[[i]][[2]]), with=FALSE]
                 list(tmp, name)
               }

goldMeans <- as.data.frame(out[[1]])
colnames(goldMeans) <- unlist(out[[2]])

ts2 <- cbind(ts2, goldMeans)
rm(goldMeans)
gc()


goldFeats2 <- 100
feat_gold <- gold_featuresUnCor(featCorDF, goldFeats2)

# Do not parallelize -- too much memory for some reason
set.seed(136)
out <- foreach(i=1:length(feat_gold), .combine='comb', .multicombine=TRUE,
               .init=list(list(), list()), .packages=c("data.table")) %dorng% {
                 name <- paste0(feat_gold[[i]][[1]],"_",feat_gold[[i]][[2]],"_corAdd")
                 tmp <- ts2[,as.character(feat_gold[[i]][[1]]), with=FALSE] + ts2[,as.character(feat_gold[[i]][[2]]), with=FALSE]
                 list(tmp, name)
               }

goldAdds <- as.data.frame(out[[1]])
colnames(goldAdds) <- unlist(out[[2]])

ts2 <- cbind(ts2, goldAdds)
rm(goldAdds)
gc()

if (length(c(hc,as.character(featCorDF$V2[1:goldFeats])))>0)
  ts2 <- ts2[,-c(hc,as.character(featCorDF$V2[1:goldFeats])),with=FALSE]
######################################################


############
## Helper columns
############
# Scale variables so a few don't overpower the helper columns
pp <- preProcess(ts2[filter==0,-excludeCols,with=FALSE], method=c("zv","center","scale","medianImpute"))
ts2 <- predict(pp, ts2)

summ <- as.data.frame(ts2[ts2$filter==0, colnames(ts2) %in% c("target",numCols),with=FALSE] %>% group_by(target) %>%
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
  ts2[[ncol(ts2)+1]] <- rowSums(ts2[,helpCols[[i+1]], with=FALSE])
  colnames(ts2)[ncol(ts2)] <- paste0("X", i, "_helper")
}

##################
## Create summary variables for high-dimensional factors
##################
factorCols <- colnames(ts2)[sapply(ts2, is.factor)]
highCardFacts <- colnames(ts2[,factorCols,with=FALSE])[sapply(ts2[,factorCols,with=FALSE], function(x) length(unique(x))>30)]

for(ii in highCardFacts) {
  print(ii)
  x <- data.frame(x1=ts2[, ii,with=FALSE])
  x[,ii] <- as.numeric(x[,ii])
  ts2[, paste(ii, "_num", sep="")] <- x
}


for(ii in highCardFacts) {
  print(ii)
  x <- data.frame(x1=ts2[, ii,with=FALSE])
  colnames(x) <- "x1"
  x$x1 <- as.numeric(x$x1)
  sum1 <- sqldf("select x1, sum(1) as cnt
                from x  group by 1 ")
  tmp <- sqldf("select cnt from x a left join sum1 b on a.x1=b.x1")
  ts2[, paste(ii, "_cnt", sep="")] <- tmp$cnt
}

# Replace high cardinality factors with target mean
for(ii in highCardFacts) {
  name <- paste0(ii, "_targetMean")
  ts2[,name] <- catNWayAvgCV(data = ts2, c(ii, "dummy"), y = "target",pred0 = "pred0",filter = ts2$filter==0, k = 0, f = 0, r_k = 0.00, cv=cvFoldsList)
}


ts2 <- ts2[,!colnames(ts2) %in% highCardFacts,with=FALSE]

##################
## Create dummy variables for low-dimensional factors
##################

dummy <- dummyVars( ~. -1, data = ts2[,-c("dummy","pred0"),with=FALSE])
ts2 <- data.frame(predict(dummy, ts2))

# varnames <- c(names(ts2[, !colnames(ts2) %in% c("ID","target","filter","dummy","pred0")]))
# 
# ts2_pca <- preProcess(ts2[,varnames], method = c("pca","center","scale"), pcaComp=100)
# ts2_pca_feats <- predict(ts2_pca, ts2[,varnames])
# 
# set.seed(201601)
# tsne_feats <- Rtsne(data.matrix(ts2_pca_feats), dims=2, initial_dims = 100, perplexity=30, theta=0.1, pca=FALSE, check_duplicates=FALSE, max_iter=500, verbose=TRUE)
# tsne_Y <- as.data.frame(tsne_feats$Y)
# colnames(tsne_Y) <- c("tsne_1", "tsne_2")
# tsne_Y <- cbind(ID=ts2$ID, tsne_Y)
# write.csv(tsne_Y, "./data_trans/tsne_v24.csv", row.names=FALSE)
# tsne_Y$target <- as.factor(make.names(ts2$target))
# (gg <- ggplot(tsne_Y[ts2$filter==0,], aes(x=tsne_1, y=tsne_2, colour=target)) + geom_point(size=1))
# 
# ts2<- cbind(ts2, tsne_Y[,2:3])

###################
## Write CSV file
###################
write.csv(as.data.frame(helpCols), "./data_trans/helpCols_v30.csv", row.names=FALSE)
save(helpCols, file="./data_trans/helpCols_v30.rda")

ts2 <- ts2[order(ts2$filter, ts2$ID),]
write_csv(ts2, "./data_trans/ts2Trans_v30.csv")
