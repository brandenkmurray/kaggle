library(data.table)
library(ggplot2)
setwd("/home/branden/Documents/kaggle/walmart")
xgb5preds <- read.csv("./stack_models/cvPreds_xgb5.csv")
names(xgb5preds) <- gsub(pattern = "xgb5_TripType_",replacement = "",x = names(xgb5preds))


t1 <- data.table(read.csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class,TripType)]

# t1$TripType <- paste0("xgb5_TripType_",t1$TripType)

df <- cbind(xgb5preds, TripType=t1$TripType)
df$VisitNumber <- NULL

xMelt <- melt(df, measure.vars=1:38)


(pl <- ggplot(data=xMelt, aes(x=TripType, y=value)) +
  geom_point(aes(alpha=0.3),stat = "identity", position="jitter",size=0.1) +
  facet_grid(variable ~ TripType, scales="free")
)



train <- data.table(read.csv("train.csv"))
View(train[train$TripType %in% c("38","39"),])
library(plyr)
library(dplyr)
ts1Trans <- read.csv("ts1Trans.csv")
View(ts1Trans[ts1Trans$class %in% c("30","31"),])
ts1Trans_7_8_9 <- ts1Trans[ts1Trans$class %in% c("30","31"),1:115]

sum_38_39 <- ts1Trans_38_39 %>% group_by(class) %>% 
  summarise_each(funs(mean))
    
  


aggregate( ts1Trans_38_39[,4:116,with=FALSE], df[,2,with=FALSE], FUN = mean )
plyr::ddply(ts1Trans_38_39, .(class), numcolwise(sum))


ts1Trans2 <- read.csv("ts1Trans.csv")
View(ts1Trans2[ts1Trans2$class %in% c("4","5","6"),16:115])
ts1Trans_7_8_9 <- ts1Trans2[ts1Trans2$class %in% c("4","5","6"),1:115]

sum_7_8_9 <- ts1Trans_7_8_9 %>% group_by(class) %>% 
  summarise_each(funs(mean))

View(sum_7_8_9)


View(ts1Trans2[ts1Trans2$class %in% c("4","27","28","31","37"),16:115])
ts1Trans_7_39 <- ts1Trans2[ts1Trans2$class %in% c("4","27","28","31","37"),1:115]

sum_7_39<- ts1Trans_7_39 %>% group_by(class) %>% 
  summarise_each(funs(mean))

View(sum_7_39)

sum <- ts1Trans2 %>% group_by(class) %>% 
  summarise_each(funs(mean))

View(sum[,16:115])
