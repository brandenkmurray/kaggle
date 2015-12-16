factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)
  
  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

NormalizedGini <- function(data, lev = NULL, model = NULL) {
  SumModelGini <- function(solution, submission) {
    df = data.frame(solution = solution, submission = submission)
    df <- df[order(df$submission, decreasing = TRUE),]
    df$random = (1:nrow(df))/nrow(df)
    totalPos <- sum(df$solution)
    df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
    df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
    df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
    return(sum(df$Gini))
  }
  
  solution=data$obs
  submission=data$pred
  result=SumModelGini(solution, submission) / SumModelGini(solution, solution)
  names(result) <- "Gini"
  result
}

my.f2cnt<-function(th2, vn1, vn2, filter=TRUE) {
  df<-data.frame(f1=th2[,vn1], f2=th2[,vn2], filter=filter)
  sum1<-sqldf("select f1, f2, count(*) as cnt from df where filter=1 group by 1,2")
  tmp<-sqldf("select b.cnt from df a left join sum1 b on a.f1=b.f1 and a.f2=b.f2")
  tmp$cnt[is.na(tmp$cnt)]<-0
  return(tmp$cnt)
}

#shrank and randomized leave-one-out average actual for categorical variables 
my_exp2<-function(d1, vn1, vn2, y, vnp, filter, cred_k, r_k=0.3){
  d2<-d1[, c(vn1, vn2, y, vnp)]
  names(d2)<-c("f1", "f2", "a", "p")
  d2$filter<-filter
  sum1<-sqldf("select f1, f2, sum(1) as cnt, sum(p) as sump, sum(a) as suma from d2 where filter=1 group by 1,2")
  tmp1<-sqldf("select a.p, b.cnt, b.sump, b.suma from d2 a left join sum1 b on a.f1=b.f1 and a.f2=b.f2")
  tmp1$cnt[is.na(tmp1$cnt)]<-0
  tmp1$avgp<-with(tmp1, sump/cnt)
  tmp1$avgp[is.na(tmp1$avgp)]<-0
  tmp1$suma[is.na(tmp1$suma)]<-0
  # subtract 1 from the counts in the training variables
  tmp1$cnt[filter]<-tmp1$cnt[filter]-1
  # subtract the response variable from the observation from the sum of actions associated with the 2 way interatction in that observation
  tmp1$suma[filter]<-tmp1$suma[filter]-d1$y[filter]
  # By using the filter in the previous steps, the same calculations can be applied to the train and test data in the next steps
  tmp1$exp_a<-with(tmp1, suma/cnt)
  tmp1$adj_a<-with(tmp1, (suma+p*cred_k)/(cnt+cred_k))
  tmp1$exp_a[is.na(tmp1$exp_a)]<-tmp1$p[is.na(tmp1$exp_a)]
  tmp1$adj_a[is.na(tmp1$adj_a)]<-tmp1$p[is.na(tmp1$adj_a)]
  tmp1$adj_a[filter]<-tmp1$adj_a[filter]*(1+(runif(sum(filter))-0.5)*r_k)
  return(tmp1)
}