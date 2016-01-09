library(readr)
library(data.table)
library(xgboost)
library(caretEnsemble)
library(reshape2)
library(dplyr)
library(proxy)
library(qlcMatrix)
library(cccd)
library(igraph)

setwd("/home/branden/Documents/kaggle/walmart")

t1 <- data.table(read.csv("train.csv"))
s1 <- data.table(read.csv("test.csv"))

tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1$TripType <- NULL

s1$class <- -1
t1$filter <- 0
s1$filter <- 2


# list of top UPCs
topUPC <- names(sort(which(table(t1$Upc)>10), decreasing=TRUE))
t1$Upc2 <- ifelse(t1$Upc %in% topUPC, t1$Upc, "Other")
s1$Upc2 <- ifelse(s1$Upc %in% topUPC, s1$Upc, "Other")

x2 <- dcast.data.table(t1, Upc2 ~ class, fun.aggregate=length, value.var = "ScanCount")
x3 <- x2[,2:ncol(x2),with=FALSE]/rowSums(x2[,2:ncol(x2), with=FALSE])
x3 <- cbind(x2$Upc2,x3)

rowMax <- apply(x3[,2:ncol(x3), with=FALSE], 1, max)
topUPC <- x3$V1[which(rowMax>.60)]

topUPC2 <- names(sort(table(t1$Upc), decreasing=TRUE))[1:2000]
upcList <- unique(c(topUPC, topUPC2))


t1$Upc2 <- ifelse(t1$Upc %in% upcList, t1$Upc, "Other")
s1$Upc2 <- ifelse(s1$Upc %in% upcList, s1$Upc, "Other")

ts1 <- rbind(t1, s1)
ts1[is.na(ts1)] <- -99

ts1$Returns <- -ts1$ScanCount
ts1$Returns[ts1$Returns < 0] <- 0
ts1$Purchases <- ts1$ScanCount
ts1$Purchases[ts1$Purchases < 0] <- 0


rm(t1)
rm(s1)
gc()

entropy <- function(x) {
  tab <- table(as.character(x))
  e <- sum(log(seq(1,sum(tab))))
  for (i in tab){
    e <- e - sum(log(seq(1,i)))
  }
  return(e)
}

entropy2 <- function(x, count) {
  tmp <- data.frame(x=x, count=count)
  tmp <- tmp[tmp$count>0,]
  if (nrow(tmp)==0)
  {return(0)}
  else {
    tab <- aggregate(count ~ x, tmp, sum)
    e <- sum(log(seq(1,sum(tab$count))))
    for (i in tab$count){
      e <- e - sum(log(seq(1,i)))
    }
    return(e)
  }
}


data_transform <- function(data){
  data$ScanCount <- as.numeric(data$ScanCount)
  data$VisitNumber <- as.factor(data$VisitNumber)
  data$FinelineNumber <- as.factor(data$FinelineNumber)
  x <- data[, list(n=length(DepartmentDescription),
                   uniqDept=length(unique(DepartmentDescription)),
                   uniqFine=length(unique(FinelineNumber)),
                   uniqUpc=length(unique(Upc)),
                   deptEntropy=entropy(DepartmentDescription),
                   fineEntropy=entropy(FinelineNumber),
                   upcEntropy=entropy(Upc),
                   deptEntropy2=entropy2(DepartmentDescription, Purchases),
                   fineEntropy2=entropy2(FinelineNumber, Purchases),
                   upcEntropy2=entropy2(Upc, Purchases),
                   purchases = sum(Purchases),
                   returns = sum(Returns),
                   purchDepts = length(unique(DepartmentDescription[Purchases>0])),
                   returnDepts = length(unique(DepartmentDescription[Returns>0])),
                   purchFine = length(unique(FinelineNumber[Purchases>0])),
                   returnFine = length(unique(FinelineNumber[Returns>0])),
                   purchUpc = length(unique(Upc[Purchases>0])),
                   returnUpc = length(unique(Upc[Returns>0])),
                   netScans=sum(Purchases + Returns),
                   maxScans=max(ScanCount),
                   minScans=min(ScanCount),
                   meanScans=mean(ScanCount),
                   medScans = median(ScanCount)
                   #                    modeScans = names(sort(-table(ScanCount)))[1],
                   #                    modeDept = names(sort(-table(DepartmentDescription)))[1],
                   #                    modeFine = names(sort(-table(FinelineNumber)))[1],
                   #                    modeUpc = names(sort(-table(Upc)))[1]
  ), by=list(VisitNumber,class,filter)]
  x <- x[, ':='(fineDeptRatio=uniqFine/uniqDept,
                upcDeptRatio=uniqUpc/uniqDept,
                upcFineRatio=uniqUpc/uniqFine,
                returnRatio = returns / netScans,
                deptFineEntRatio=ifelse(is.infinite(deptEntropy/fineEntropy),0,deptEntropy/fineEntropy),
                deptUpcEntRatio=ifelse(is.infinite(deptEntropy/upcEntropy),0,deptEntropy/upcEntropy),
                fineUpcEntRatio=ifelse(is.infinite(fineEntropy/upcEntropy),0,fineEntropy/upcEntropy),
                deptFineEntRatio2=ifelse(is.infinite(deptEntropy2/fineEntropy2),0,deptEntropy2/fineEntropy2),
                deptUpcEntRatio2=ifelse(is.infinite(deptEntropy2/upcEntropy2),0,deptEntropy2/upcEntropy2),
                fineUpcEntRatio2=ifelse(is.infinite(fineEntropy2/upcEntropy2),0,fineEntropy2/upcEntropy2),
                scansDeptRatio=netScans/uniqDept,
                scansFineRatio=netScans/uniqFine,
                scansUpcRatio=netScans/uniqUpc)]
  
  xWeekday <- dcast.data.table(VisitNumber~Weekday, value.var="Purchases",
                               fun.aggregate = sum, data=data)
  xWeekday <- data.table(xWeekday[,"VisitNumber",with=FALSE], prop.table(as.matrix(xWeekday[,2:ncol(xWeekday), with=FALSE]),margin=1))
  xDept <- dcast.data.table(VisitNumber~DepartmentDescription, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  xDept <- data.table(xDept[,"VisitNumber",with=FALSE], prop.table(as.matrix(xDept[,2:ncol(xDept), with=FALSE]),margin=1))
  xFine <- dcast.data.table(VisitNumber~FinelineNumber, value.var="Purchases",
                            fun.aggregate = sum, data=data)
  xFine <- data.table(xFine[,"VisitNumber",with=FALSE], prop.table(as.matrix(xFine[,2:ncol(xFine), with=FALSE]),margin=1))
  xUpc <- dcast.data.table(VisitNumber~Upc2, value.var="Purchases",
                           fun.aggregate = sum, data=data)
  xUpc <- data.table(xUpc[,"VisitNumber",with=FALSE], prop.table(as.matrix(xUpc[,2:ncol(xUpc), with=FALSE]),margin=1))
  
  xAgg <- merge(x, xWeekday, by="VisitNumber")
  xAgg <- merge(xAgg, xDept, by="VisitNumber")
  xAgg <- merge(xAgg, xFine, by="VisitNumber")
  xAgg <- merge(xAgg, xUpc, by="VisitNumber")
  return(xAgg)
}

ts1Trans4 <- data_transform(ts1)
# Reorder the data set so that the train set is at the top and then order by VisitNumber
ts1Trans4 <- ts1Trans4[order(filter, VisitNumber),]
# Some entropy values were 0. This created some NAs for the entropy ratios
ts1Trans4[is.na(ts1Trans4)] <- 0

ts1Trans4[ , TripType38_helper := sum(`GROCERY DRY GOODS`, DAIRY, `COMM BREAD`,`PRE PACKED DELI`, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType39_helper := sum(`PETS AND SUPPLIES`  ,`PERSONAL CARE`  ,`HOUSEHOLD CHEMICALS/SUPP`  ,BEAUTY  ,`PHARMACY OTC`, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType7_helper := sum(BAKERY  ,`COMM BREAD`  ,DAIRY  ,`DSD GROCERY`  ,`FROZEN FOODS`  ,`GROCERY DRY GOODS`  ,`MEAT - FRESH & FROZEN`  ,`PRE PACKED DELI`  ,PRODUCE    ,`SERVICE DELI`, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType8_helper := sum(DAIRY  ,`DSD GROCERY`  ,`PERSONAL CARE`  ,BEAUTY  ,`GROCERY DRY GOODS`  ,`IMPULSE MERCHANDISE`  ,PRODUCE, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType9_helper := sum(AUTOMOTIVE  ,CELEBRATION  ,`MENS WEAR`  ,`OFFICE SUPPLIES`, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType35_helper := sum(`CANDY, TOBACCO, COOKIES`  ,`DSD GROCERY`  ,`IMPULSE MERCHANDISE`, na.rm=TRUE), by=1:NROW(ts1Trans4)]
ts1Trans4[ , TripType36_helper := sum(BEAUTY  ,`PERSONAL CARE`  ,`PHARMACY OTC`  ,`PETS AND SUPPLIES`, na.rm=TRUE), by=1:NROW(ts1Trans4)]


write_csv(ts1Trans4, "./data_trans/ts1Trans4_prop.csv")

# Similarity matrices
# Use only train data to get similarities
ts1_dept <- as.matrix(ts1Trans4[, 47:115, with=FALSE])
ts1_dept_Matrix <- Matrix(ts1_dept)
ts1_cosSparse <- as.matrix(cosSparse(ts1_dept_Matrix))
ts1_dist <- pr_simil2dist(ts1_cosSparse)
ts1_dist_nng <- nng(dx=ts1_dist, k=5)

V(ts1_dist_nng)$name <- rownames(ts1_cosSparse)
E(ts1_dist_nng)$weight <- apply(get.edges(ts1_dist_nng,1:ecount(ts1_dist_nng)),1,function(x)ts1_cosSparse[x[1],x[2]])

ts1_dist_adj <- as_adjacency_matrix(ts1_dist_nng, attr="weight")
ts1_dist_adj_mat <- as.matrix(ts1_dist_adj)
dept_diag <- diag(x=1, nrow=nrow(ts1_dist_adj_mat))
ts1_dist_adj_mat <- ts1_dist_adj_mat + dept_diag

ts1_dist_adj_mat <- ts1_dist_adj_mat %*% diag(1/rowSums(ts1_dist_adj_mat))

ts1_dept_simil <- as.data.frame(ts1_dept  %*% ts1_dist_adj_mat)
colnames(ts1_dept_simil) <- colnames(ts1_dept)

write_csv(ts1_dept_simil, "./data_trans/ts1v4_dept_simil_prop.csv")


# Fineline distance/similarity
# Use only train data to get similarities
ts1_fine <- as.matrix(ts1Trans4[, 116:5469, with=FALSE])
ts1_fine_Matrix <- Matrix(ts1_fine)
ts1_cosSparse <- as.matrix(cosSparse(ts1_fine_Matrix))
ts1_dist <- pr_simil2dist(ts1_cosSparse)
ts1_dist_nng <- nng(dx=ts1_dist, k=5)

V(ts1_dist_nng)$name <- rownames(ts1_cosSparse)
E(ts1_dist_nng)$weight <- apply(get.edges(ts1_dist_nng,1:ecount(ts1_dist_nng)),1,function(x)ts1_cosSparse[x[1],x[2]])

ts1_dist_adj <- as_adjacency_matrix(ts1_dist_nng, attr="weight")
ts1_dist_adj_mat <- as.matrix(ts1_dist_adj)
fine_diag <- diag(x=1, nrow=nrow(ts1_dist_adj_mat))
ts1_dist_adj_mat <- ts1_dist_adj_mat + fine_diag

ts1_dist_adj_mat <- ts1_dist_adj_mat %*% diag(1/rowSums(ts1_dist_adj_mat))

ts1_fine_simil <- as.data.frame(ts1_fine %*% ts1_dist_adj_mat)
colnames(ts1_fine_simil) <- colnames(ts1_fine)

write_csv(ts1_fine_simil, "./data_trans/ts1v4_fine_simil_prop.csv")


# Fineline distance/similarity
# Use only train data to get similarities
ts1_upc <- as.matrix(ts1Trans4[, 5470:8273, with=FALSE])
ts1_upc_Matrix <- Matrix(ts1_upc)
ts1_cosSparse <- as.matrix(cosSparse(ts1_upc_Matrix))
ts1_dist <- pr_simil2dist(ts1_cosSparse)
ts1_dist_nng <- nng(dx=ts1_dist, k=5)

V(ts1_dist_nng)$name <- rownames(ts1_cosSparse)
E(ts1_dist_nng)$weight <- apply(get.edges(ts1_dist_nng,1:ecount(ts1_dist_nng)),1,function(x)ts1_cosSparse[x[1],x[2]])

ts1_dist_adj <- as_adjacency_matrix(ts1_dist_nng, attr="weight")
ts1_dist_adj_mat <- as.matrix(ts1_dist_adj)
upc_diag <- diag(x=1, nrow=nrow(ts1_dist_adj_mat))
ts1_dist_adj_mat <- ts1_dist_adj_mat + upc_diag

ts1_dist_adj_mat <- ts1_dist_adj_mat %*% diag(1/rowSums(ts1_dist_adj_mat))

ts1_upc_simil <- as.data.frame(ts1_upc %*% ts1_dist_adj_mat)
colnames(ts1_upc_simil) <- colnames(ts1_upc)

write_csv(ts1_upc_simil, "./data_trans/ts1v4_upc_simil_prop.csv")

# Replace with similarity 
ts1Trans4[, colnames(ts1_dept_simil)] <- ts1_dept_simil
ts1Trans4[, colnames(ts1_fine_simil)] <- ts1_fine_simil
ts1Trans4[, colnames(ts1_upc_simil)] <- ts1_upc_simil
write_csv(ts1Trans4, "./data_trans/ts1Trans4_prop_simil.csv")
