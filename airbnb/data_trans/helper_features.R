library(data.table)
library(ggplot2)
setwd("/home/branden/Documents/kaggle/airbnb")
xgb4preds <- read.csv("./stack_models/cvPreds_xgb4.csv")

t1 <- data.table(read.csv("train.csv"))
tripClasses <- data.frame(TripType=sort(unique(t1$TripType)), class=seq(0,37))
t1 <- merge(t1, tripClasses, by="TripType")
t1 <- t1[order(t1$VisitNumber),]
TripType <- t1$TripType
t1 <- t1[,length(DepartmentDescription),by=list(VisitNumber,class,TripType)]

t1 <- data.table(read.csv("./train_users_2.csv"))
destClass <- data.frame(country_destination=sort(unique(t1$country_destination)), class=seq(0,11))
t1 <- merge(t1, destClass, by="country_destination")
t1 <- t1[order(t1$timestamp_first_active),]
country_destination <- t1$country_destination

# t1$TripType <- paste0("xgb5_TripType_",t1$TripType)

df <- cbind(xgb4preds, country=t1$country_destination)
df$id <- NULL

xMelt <- melt(df, measure.vars=1:12)


(pl <- ggplot(data=xMelt, aes(x=country, y=value)) +
  geom_point(aes(alpha=0.3),stat = "identity", position="jitter",size=0.1) +
  facet_grid(variable ~ country, scales="free_x")
)



train <- data.table(read.csv("train.csv"))
View(train[train$TripType %in% c("38","39"),])
library(plyr)
library(dplyr)
ts1Trans <- data.table(read.csv("./data_trans/ts1_pp_v3.csv"))
# View(ts1Trans[ts1Trans$class %in% c("30","31"),])
# ts1Trans_7_8_9 <- ts1Trans[ts1Trans$class %in% c("30","31"),1:115]

summ <- ts1Trans[filter==0,2:ncol(ts1Trans),with=FALSE] %>% group_by(class) %>% 
  summarise_each(funs(mean))

View(summ)

mn1 <- sapply(summ[,3:ncol(summ),with=FALSE], mean)
sd1 <- sapply(summ[,3:ncol(summ),with=FALSE], sd)

hi <- mn1+2*sd1
lo <- mn1-2*sd1

tmp <- summ[class==0,3:ncol(summ), with=FALSE] > hi
whichHi <- which(tmp)
tmp <- as.data.frame(tmp)
colnames(tmp[,whichHi])


View(ts1Trans[filter==0 & class==0, colnames(tmp[,whichHi]), with=FALSE])

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
