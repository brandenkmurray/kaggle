library(data.table)

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
leak <- fread("./data_trans/data_trans_leak_alltrain_wtdprops.csv", key="id2")
leakTopK <- as.data.table(t(apply(leak[,2:ncol(leak),with=F], 1, function(x) names(sort(x[x>0], decreasing=T)[1:10]))))
colnames(leakTopK)[1:ncol(leakTopK)] <- paste0("leakTop_",1:(ncol(leakTopK)))
for (col in 1:ncol(leakTopK)){
  set(leakTopK, j=col, value = as.numeric(str_extract(leakTopK[[col]], "[0-9]{1,2}$")))
}
for (col in 1:ncol(leakTopK)){
  set(leakTopK, i=which(is.na(leakTopK[[col]])), j=col, value = -1)
}
leakTopK <- cbind(id2=leak$id2, leakTopK)
setkey(leakTopK, id2)
ts1[leakTopK, (colnames(leakTopK)[2:ncol(leakTopK)]):=leakTopK[,2:ncol(leakTopK),with=FALSE]]; rm(leak, leakTopK); gc()

destid_clust <- fread("./data_trans/data_trans_destid_alltrain_wtdprops.csv", key="id2")
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