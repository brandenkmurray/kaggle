require(irlba)
require(sqldf)
require(gbm)
require(randomForest)
require(extraTrees)
require(glmnet)
library(doParallel)
library(e1071)
library(qdapTools)
library(caret)

setwd("/home/branden/Documents/kaggle/libmut")
t1 <- read.csv("train.csv")
s1 <- read.csv("test.csv")

s1$Hazard <- (-1)
ts1 <- rbind(t1, s1)

ts1$y <- ts1$Hazard



ts1$dummy <- 1
ts1$split1 <- 0
ts1$split1[ts1$Hazard < 0] <-2
ts1$pred0<-with(ts1[ts1$split1==0,], mean(Hazard*1.0))

#######################################################################################################################
#one way count
mean_t<-with(ts1[ts1$split1==0,], sum(Hazard)*1.0/length(Hazard))
for(ii in 3:34) {
  print(names(ts1)[ii])
  ts1$x<-ts1[, ii]
  sum1<-sqldf("select x, sum(1) as cnt
              from ts1  group by 1 ")
  tmp<-sqldf("select cnt from ts1 a left join sum1 b on a.x=b.x")
  ts1[, paste(names(ts1)[ii], "_cnt", sep="")]<-tmp$cnt
}

###################################################################################################################
# 2-way counts
ts1$t1v4_t1v5_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V5")
ts1$t1v4_t1v6_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V6")
ts1$t1v4_t1v7_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V7")
ts1$t1v4_t1v8_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V8")
ts1$t1v4_t1v9_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V9")
ts1$t1v4_t1v11_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V11")
ts1$t1v4_t1v12_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V12")
ts1$t1v4_t1v15_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V15")
ts1$t1v4_t1v16_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V16")
ts1$t1v4_t1v17_cnt <- my.f2cnt(ts1, "T1_V4", "T1_V17")
ts1$t1v4_t2v3_cnt <- my.f2cnt(ts1, "T1_V4", "T2_V3")
ts1$t1v4_t2v5_cnt <- my.f2cnt(ts1, "T1_V4", "T2_V5")
ts1$t1v4_t2v11_cnt <- my.f2cnt(ts1, "T1_V4", "T2_V11")
ts1$t1v4_t2v12_cnt <- my.f2cnt(ts1, "T1_V4", "T2_V12")
ts1$t1v4_t2v13_cnt <- my.f2cnt(ts1, "T1_V4", "T2_V13")

ts1$t1v5_t1v6_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V6")
ts1$t1v5_t1v7_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V7")
ts1$t1v5_t1v8_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V8")
ts1$t1v5_t1v9_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V9")
ts1$t1v5_t1v11_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V11")
ts1$t1v5_t1v12_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V12")
ts1$t1v5_t1v15_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V15")
ts1$t1v5_t1v16_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V16")
ts1$t1v5_t1v17_cnt <- my.f2cnt(ts1, "T1_V5", "T1_V17")
ts1$t1v5_t2v3_cnt <- my.f2cnt(ts1, "T1_V5", "T2_V3")
ts1$t1v5_t2v5_cnt <- my.f2cnt(ts1, "T1_V5", "T2_V5")
ts1$t1v5_t2v11_cnt <- my.f2cnt(ts1, "T1_V5", "T2_V11")
ts1$t1v5_t2v12_cnt <- my.f2cnt(ts1, "T1_V5", "T2_V12")
ts1$t1v5_t2v13_cnt <- my.f2cnt(ts1, "T1_V5", "T2_V13")

ts1$t1v6_t1v7_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V7")
ts1$t1v6_t1v8_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V8")
ts1$t1v6_t1v9_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V9")
ts1$t1v6_t1v11_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V11")
ts1$t1v6_t1v12_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V12")
ts1$t1v6_t1v15_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V15")
ts1$t1v6_t1v16_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V16")
ts1$t1v6_t1v17_cnt <- my.f2cnt(ts1, "T1_V6", "T1_V17")
ts1$t1v6_t2v3_cnt <- my.f2cnt(ts1, "T1_V6", "T2_V3")
ts1$t1v6_t2v5_cnt <- my.f2cnt(ts1, "T1_V6", "T2_V5")
ts1$t1v6_t2v11_cnt <- my.f2cnt(ts1, "T1_V6", "T2_V11")
ts1$t1v6_t2v12_cnt <- my.f2cnt(ts1, "T1_V6", "T2_V12")
ts1$t1v6_t2v13_cnt <- my.f2cnt(ts1, "T1_V6", "T2_V13")

ts1$t1v7_t1v8_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V8")
ts1$t1v7_t1v9_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V9")
ts1$t1v7_t1v11_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V11")
ts1$t1v7_t1v12_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V12")
ts1$t1v7_t1v15_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V15")
ts1$t1v7_t1v16_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V16")
ts1$t1v7_t1v17_cnt <- my.f2cnt(ts1, "T1_V7", "T1_V17")
ts1$t1v7_t2v3_cnt <- my.f2cnt(ts1, "T1_V7", "T2_V3")
ts1$t1v7_t2v5_cnt <- my.f2cnt(ts1, "T1_V7", "T2_V5")
ts1$t1v7_t2v11_cnt <- my.f2cnt(ts1, "T1_V7", "T2_V11")
ts1$t1v7_t2v12_cnt <- my.f2cnt(ts1, "T1_V7", "T2_V12")
ts1$t1v7_t2v13_cnt <- my.f2cnt(ts1, "T1_V7", "T2_V13")

ts1$t1v8_t1v9_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V9")
ts1$t1v8_t1v11_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V11")
ts1$t1v8_t1v12_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V12")
ts1$t1v8_t1v15_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V15")
ts1$t1v8_t1v16_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V16")
ts1$t1v8_t1v17_cnt <- my.f2cnt(ts1, "T1_V8", "T1_V17")
ts1$t1v8_t2v3_cnt <- my.f2cnt(ts1, "T1_V8", "T2_V3")
ts1$t1v8_t2v5_cnt <- my.f2cnt(ts1, "T1_V8", "T2_V5")
ts1$t1v8_t2v11_cnt <- my.f2cnt(ts1, "T1_V8", "T2_V11")
ts1$t1v8_t2v12_cnt <- my.f2cnt(ts1, "T1_V8", "T2_V12")
ts1$t1v8_t2v13_cnt <- my.f2cnt(ts1, "T1_V8", "T2_V13")

ts1$t1v9_t1v11_cnt <- my.f2cnt(ts1, "T1_V9", "T1_V11")
ts1$t1v9_t1v12_cnt <- my.f2cnt(ts1, "T1_V9", "T1_V12")
ts1$t1v9_t1v15_cnt <- my.f2cnt(ts1, "T1_V9", "T1_V15")
ts1$t1v9_t1v16_cnt <- my.f2cnt(ts1, "T1_V9", "T1_V16")
ts1$t1v9_t1v17_cnt <- my.f2cnt(ts1, "T1_V9", "T1_V17")
ts1$t1v9_t2v3_cnt <- my.f2cnt(ts1, "T1_V9", "T2_V3")
ts1$t1v9_t2v5_cnt <- my.f2cnt(ts1, "T1_V9", "T2_V5")
ts1$t1v9_t2v11_cnt <- my.f2cnt(ts1, "T1_V9", "T2_V11")
ts1$t1v9_t2v12_cnt <- my.f2cnt(ts1, "T1_V9", "T2_V12")
ts1$t1v9_t2v13_cnt <- my.f2cnt(ts1, "T1_V9", "T2_V13")

ts1$t1v11_t1v12_cnt <- my.f2cnt(ts1, "T1_V11", "T1_V12")
ts1$t1v11_t1v15_cnt <- my.f2cnt(ts1, "T1_V11", "T1_V15")
ts1$t1v11_t1v16_cnt <- my.f2cnt(ts1, "T1_V11", "T1_V16")
ts1$t1v11_t1v17_cnt <- my.f2cnt(ts1, "T1_V11", "T1_V17")
ts1$t1v11_t2v3_cnt <- my.f2cnt(ts1, "T1_V11", "T2_V3")
ts1$t1v11_t2v5_cnt <- my.f2cnt(ts1, "T1_V11", "T2_V5")
ts1$t1v11_t2v11_cnt <- my.f2cnt(ts1, "T1_V11", "T2_V11")
ts1$t1v11_t2v12_cnt <- my.f2cnt(ts1, "T1_V11", "T2_V12")
ts1$t1v11_t2v13_cnt <- my.f2cnt(ts1, "T1_V11", "T2_V13")


ts1$t1v12_t1v15_cnt <- my.f2cnt(ts1, "T1_V12", "T1_V15")
ts1$t1v12_t1v16_cnt <- my.f2cnt(ts1, "T1_V12", "T1_V16")
ts1$t1v12_t1v17_cnt <- my.f2cnt(ts1, "T1_V12", "T1_V17")
ts1$t1v12_t2v3_cnt <- my.f2cnt(ts1, "T1_V12", "T2_V3")
ts1$t1v12_t2v5_cnt <- my.f2cnt(ts1, "T1_V12", "T2_V5")
ts1$t1v12_t2v11_cnt <- my.f2cnt(ts1, "T1_V12", "T2_V11")
ts1$t1v12_t2v12_cnt <- my.f2cnt(ts1, "T1_V12", "T2_V12")
ts1$t1v12_t2v13_cnt <- my.f2cnt(ts1, "T1_V12", "T2_V13")

ts1$t1v15_t1v16_cnt <- my.f2cnt(ts1, "T1_V15", "T1_V16")
ts1$t1v15_t1v17_cnt <- my.f2cnt(ts1, "T1_V15", "T1_V17")
ts1$t1v15_t2v3_cnt <- my.f2cnt(ts1, "T1_V15", "T2_V3")
ts1$t1v15_t2v5_cnt <- my.f2cnt(ts1, "T1_V15", "T2_V5")
ts1$t1v15_t2v11_cnt <- my.f2cnt(ts1, "T1_V15", "T2_V11")
ts1$t1v15_t2v12_cnt <- my.f2cnt(ts1, "T1_V15", "T2_V12")
ts1$t1v15_t2v13_cnt <- my.f2cnt(ts1, "T1_V15", "T2_V13")

ts1$t1v16_t1v17_cnt <- my.f2cnt(ts1, "T1_V16", "T1_V17")
ts1$t1v16_t2v3_cnt <- my.f2cnt(ts1, "T1_V16", "T2_V3")
ts1$t1v16_t2v5_cnt <- my.f2cnt(ts1, "T1_V16", "T2_V5")
ts1$t1v16_t2v11_cnt <- my.f2cnt(ts1, "T1_V16", "T2_V11")
ts1$t1v16_t2v12_cnt <- my.f2cnt(ts1, "T1_V16", "T2_V12")
ts1$t1v16_t2v13_cnt <- my.f2cnt(ts1, "T1_V16", "T2_V13")

ts1$t1v17_t2v3_cnt <- my.f2cnt(ts1, "T1_V17", "T2_V3")
ts1$t1v17_t2v5_cnt <- my.f2cnt(ts1, "T1_V17", "T2_V5")
ts1$t1v17_t2v11_cnt <- my.f2cnt(ts1, "T1_V17", "T2_V11")
ts1$t1v17_t2v12_cnt <- my.f2cnt(ts1, "T1_V17", "T2_V12")
ts1$t1v17_t2v13_cnt <- my.f2cnt(ts1, "T1_V17", "T2_V13")

ts1$t2v3_t2v5_cnt <- my.f2cnt(ts1, "T2_V3", "T2_V5")
ts1$t2v3_t2v11_cnt <- my.f2cnt(ts1, "T2_V3", "T2_V11")
ts1$t2v3_t2v12_cnt <- my.f2cnt(ts1, "T2_V3", "T2_V12")
ts1$t2v3_t2v13_cnt <- my.f2cnt(ts1, "T2_V3", "T2_V13")

ts1$t2v5_t2v11_cnt <- my.f2cnt(ts1, "T2_V5", "T2_V11")
ts1$t2v5_t2v12_cnt <- my.f2cnt(ts1, "T2_V5", "T2_V12")
ts1$t2v5_t2v13_cnt <- my.f2cnt(ts1, "T2_V5", "T2_V13")

ts1$t2v11_t2v12_cnt <- my.f2cnt(ts1, "T2_V11", "T2_V12")
ts1$t2v11_t2v13_cnt <- my.f2cnt(ts1, "T2_V11", "T2_V13")

ts1$t2v12_t2v13_cnt <- my.f2cnt(ts1, "T2_V12", "T2_V13")


###################################################################################
# create exp variables (leave one out average actual by categorical variable)
r_k=0.3

ts1$exp_t1v4 <- my_exp2(ts1, "T1_V4", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5<- my_exp2(ts1, "T1_V5", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v6 <- my_exp2(ts1, "T1_V6", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v7 <- my_exp2(ts1, "T1_V7", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v8 <- my_exp2(ts1, "T1_V8", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9 <- my_exp2(ts1, "T1_V9", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11 <- my_exp2(ts1, "T1_V11", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v12 <- my_exp2(ts1, "T1_V12", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15 <- my_exp2(ts1, "T1_V15", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16 <- my_exp2(ts1, "T1_V16", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v17 <- my_exp2(ts1, "T1_V17", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t2v3 <- my_exp2(ts1, "T2_V3", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v5 <- my_exp2(ts1, "T2_V5", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v11 <- my_exp2(ts1, "T2_V11", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v12 <- my_exp2(ts1, "T2_V12", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v13 <- my_exp2(ts1, "T2_V13", "dummy", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

# exp variables for interactions between factors with >=8 levels
ts1$exp_t1v4_t1v5 <- my_exp2(ts1, "T1_V4", "T1_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t1v9 <- my_exp2(ts1, "T1_V4", "T1_V9", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t1v11 <- my_exp2(ts1, "T1_V4", "T1_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t1v15 <- my_exp2(ts1, "T1_V4", "T1_V15", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t1v16 <- my_exp2(ts1, "T1_V4", "T1_V16", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t2v5 <- my_exp2(ts1, "T1_V4", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t2v13 <- my_exp2(ts1, "T1_V4", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v5_t1v9 <- my_exp2(ts1, "T1_V5", "T1_V9", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t1v11 <- my_exp2(ts1, "T1_V5", "T1_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t1v15 <- my_exp2(ts1, "T1_V5", "T1_V15", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t1v16 <- my_exp2(ts1, "T1_V5", "T1_V16", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t2v5 <- my_exp2(ts1, "T1_V5", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t2v13 <- my_exp2(ts1, "T1_V5", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v9_t1v11 <- my_exp2(ts1, "T1_V9", "T1_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t1v15 <- my_exp2(ts1, "T1_V9", "T1_V15", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t1v16 <- my_exp2(ts1, "T1_V9", "T1_V16", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t2v5 <- my_exp2(ts1, "T1_V9", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t2v13 <- my_exp2(ts1, "T1_V9", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v11_t1v15 <- my_exp2(ts1, "T1_V11", "T1_V15", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t1v16 <- my_exp2(ts1, "T1_V11", "T1_V16", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t2v5 <- my_exp2(ts1, "T1_V11", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t2v13 <- my_exp2(ts1, "T1_V11", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v15_t1v16 <- my_exp2(ts1, "T1_V15", "T1_V16", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t2v5 <- my_exp2(ts1, "T1_V15", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t2v13 <- my_exp2(ts1, "T1_V15", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v16_t2v5 <- my_exp2(ts1, "T1_V16", "T2_V5", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16_t2v13 <- my_exp2(ts1, "T1_V16", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t2v5_t2v13 <- my_exp2(ts1, "T2_V5", "T2_V13", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a


vars <- sapply(ts1, is.factor)
addDimTrain <- factorToNumeric(ts1[ts1$split1==0,], ts1[ts1$split1==0,], "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))
addDimTest <- factorToNumeric(ts1[ts1$split1==0,], ts1[ts1$split1==2,], "Hazard", names(vars[vars]), c("mean","median","sd","skewness","kurtosis"))

addDim <- rbind(addDimTrain, addDimTest)
# Drop medians for columsn with only two factor levels
addDim <- subset(addDim, select=-c(median_T1_V6, median_T1_V17, median_T2_V3, median_T2_V11,median_T2_V12))
ts1 <- cbind(ts1, addDim)

# T1_V6, T1_V17, T2_V3, T2_V11, T2_V12
ts1$exp_t1v4_t1v6 <- my_exp2(ts1, "T1_V4", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t1v17 <- my_exp2(ts1, "T1_V4", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t2v3 <- my_exp2(ts1, "T1_V4", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t2v11 <- my_exp2(ts1, "T1_V4", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v4_t2v12 <- my_exp2(ts1, "T1_V4", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v5_t1v6 <- my_exp2(ts1, "T1_V5", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t1v17 <- my_exp2(ts1, "T1_V5", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t2v3 <- my_exp2(ts1, "T1_V5", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t2v11 <- my_exp2(ts1, "T1_V5", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v5_t2v12 <- my_exp2(ts1, "T1_V5", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v9_t1v6 <- my_exp2(ts1, "T1_V9", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t1v17 <- my_exp2(ts1, "T1_V9", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t2v3 <- my_exp2(ts1, "T1_V9", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t2v11 <- my_exp2(ts1, "T1_V9", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v9_t2v12 <- my_exp2(ts1, "T1_V9", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v11_t1v6 <- my_exp2(ts1, "T1_V11", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t1v17 <- my_exp2(ts1, "T1_V11", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t2v3 <- my_exp2(ts1, "T1_V11", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t2v11 <- my_exp2(ts1, "T1_V11", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v11_t2v12 <- my_exp2(ts1, "T1_V11", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v15_t1v6 <- my_exp2(ts1, "T1_V15", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t1v17 <- my_exp2(ts1, "T1_V15", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t2v3 <- my_exp2(ts1, "T1_V15", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t2v11 <- my_exp2(ts1, "T1_V15", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v15_t2v12 <- my_exp2(ts1, "T1_V15", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t1v16_t1v6 <- my_exp2(ts1, "T1_V16", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16_t1v17 <- my_exp2(ts1, "T1_V16", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16_t2v3 <- my_exp2(ts1, "T1_V16", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16_t2v11 <- my_exp2(ts1, "T1_V16", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t1v16_t2v12 <- my_exp2(ts1, "T1_V16", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a

ts1$exp_t2v5_t1v6 <- my_exp2(ts1, "T2_V5", "T1_V6", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v5_t1v17 <- my_exp2(ts1, "T2_V5", "T1_V17", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v5_t2v3 <- my_exp2(ts1, "T2_V5", "T2_V3", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v5_t2v11 <- my_exp2(ts1, "T2_V5", "T2_V11", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
ts1$exp_t2v5_t2v12 <- my_exp2(ts1, "T2_V5", "T2_V12", "Hazard", "pred0", ts1$split1==0, 40, r_k=r_k)$adj_a
