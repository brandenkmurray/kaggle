idens = list()

iter = ncol(train)

for(i in 1:iter){
  iden = c()
  for(j in (i+1):iter){
    iden = c(iden,identical(train[,i],train[,j]))
  }
  idens[[i]] = iden
}

duplicates = which(sapply(idens,sum)!=0)

duplicated = list()

for(i in 1:length(duplicates)){
  duplicated[[i]] = which(idens[[duplicates[i]]]) + duplicates[i]

}

all.duplicated = unique(unlist(duplicated))
