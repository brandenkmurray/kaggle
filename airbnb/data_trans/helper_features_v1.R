library(data.table)
library(ggplot2)
library(readr)
setwd("/home/branden/Documents/kaggle/airbnb")
# xgb4preds <- read.csv("./stack_models/cvPreds_xgb4.csv")
# 
# t1 <- data.table(read.csv("./train_users_2.csv"))
# destClass <- data.frame(country_destination=sort(unique(t1$country_destination)), class=seq(0,11))
# t1 <- merge(t1, destClass, by="country_destination")
# t1 <- t1[order(t1$timestamp_first_active),]
# country_destination <- t1$country_destination


# df <- cbind(xgb4preds, country=t1$country_destination)
# df$id <- NULL
# 
# xMelt <- melt(df, measure.vars=1:12)
# 
# (pl <- ggplot(data=xMelt, aes(x=country, y=value)) +
#   geom_point(aes(alpha=0.3),stat = "identity", position="jitter",size=0.1) +
#   facet_grid(variable ~ country, scales="free_x")
# )

library(plyr)
library(dplyr)
ts1Trans <- read.csv("./data_trans/ts1_pp_v3.csv")


summ <- ts1Trans[filter==0,2:ncol(ts1Trans)] %>% group_by(class) %>% 
  summarise_each(funs(mean))

mn1 <- sapply(summ[,3:ncol(summ)], mean)
sd1 <- sapply(summ[,3:ncol(summ)], sd)

hi <- mn1+2*sd1
lo <- mn1-2*sd1

helpCols <- list()

for (i in 0:11){
  tmpHi <- (summ[summ$class==i,3:ncol(summ)] - mn1)/sd1
  hiNames <- colnames(tmpHi[,order(tmpHi)][,1:15])
  loNames <- colnames(tmpHi[,order(tmpHi,decreasing = TRUE)][,1:15])

  helpCols[[i+1]] <- c(hiNames, loNames)
  
}
names(helpCols) <- paste0("X", seq_along(helpCols)-1)

write.csv(as.data.frame(helpCols), "./data_trans/helpCols.csv")
save(helpCols, file="./data_trans/helpCols.rda")
