library(dplyr)
train <- read.csv("/media/branden/SSHD1/kaggle/bnp/train.csv")
test <- read.csv("/media/branden/SSHD1/kaggle/bnp/test.csv")
train <- data.frame(data.matrix(train))
train[is.na(train)] <- -1
test <- data.frame(data.matrix(test))
test[is.na(test)] <- -1
preds <- train %>% group_by(v31,v66,v47,v24,v79,v30,v113) %>% summarise(target=mean(target))
testPreds <- merge(test, preds, all.x=TRUE)
testPreds$target[is.na(testPreds$target)] <- mean(testPreds$target, na.rm=TRUE)
write.csv(data.frame(ID=testPreds$ID[order(testPreds$ID)], PredictedProb=testPreds$target[order(testPreds$ID)]), "/media/branden/SSHD1/kaggle/bnp/beatTheBenchmark.csv",row.names = FALSE)



sapply(train[,c("v3","v24","v30","v31","v38","v47","v52","v62","v66","v71","v72","v74","v75","v79","v91","v107","v110","v112","v113","v129")], function (x) length(unique(x)))
