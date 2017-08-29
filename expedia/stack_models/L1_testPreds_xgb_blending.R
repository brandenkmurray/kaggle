library(data.table)


xgb16 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v16_probs.csv", header=TRUE)
xgb17 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v17_probs.csv", header=TRUE)
xgb18 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v18_probs.csv", header=TRUE)
xgb19 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v19_probs.csv", header=TRUE)
xgb20 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v20_probs.csv", header=TRUE)
xgb21 <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_v21_probs.csv", header=TRUE)
xgb1_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb1_loop_probs.csv", header=TRUE)
xgb2_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb2_loop_probs.csv", header=TRUE)
xgb3_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb3_loop_probs.csv", header=TRUE)
xgb5_loop <- fread("./stack_models/L1/testPreds/L1_testPreds_xgb5_loop_probs.csv", header=TRUE)

testPreds <- (((xgb16[,2:ncol(xgb16),with=F] + xgb17[,2:ncol(xgb17),with=F] + xgb18[,2:ncol(xgb18),with=F] + xgb19[,2:ncol(xgb19),with=F] + xgb20[,2:ncol(xgb20),with=F] + xgb21[,2:ncol(xgb21),with=F] + xgb1_loop[,2:ncol(xgb1_loop),with=F] + xgb2_loop[,2:ncol(xgb2_loop),with=F]+ xgb3_loop[,2:ncol(xgb3_loop),with=F])/9) + xgb5_loop[,2:ncol(xgb5_loop),with=F])/2


colnames(testPreds) <- as.character(0:99)
# write.csv(data.frame(id=ts1Trans$id[ts1Trans$filter==2], t(testPreds)), "./stack_models/testPredsProbs_xgb11.csv", row.names=FALSE)
testPreds_top5 <- as.data.frame(t(apply(testPreds, 1, function(x) names(sort(x, decreasing=T)[1:5]))))

testPreds_top5_concat <- do.call("paste", c(testPreds_top5, sep=" "))

submission <- data.table(id=seq(0,nrow(testPreds_top5)-1,1), hotel_cluster=testPreds_top5_concat)
fwrite(submission, "./stack_models/L1/testPreds/L1_testPreds_xgb16-loop5_wtdavg.csv")
