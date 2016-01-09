library(data.table)

x <- data.frame(visit=sample(c(1,2,3,4,5,6,7,8,9,10), size=100, replace=T), scans=sample(c(-1,1,2,3,4),size=100,replace=TRUE), upc=sample(c("a","b","c","d","e","f","g","h","i","j","k","l","m","n"), size=100, replace=T))
x <- as.data.table(x[order(x$visit),])
x



x1 <- x[,
        list(sum=sum(scans),
             netScans=sum(abs(scans))),
                by=c("visit")]
x1

# NEED TO REDUCE UPCs to only those with >10 (or some other value) appearances


x2 <- dcast.data.table(x, upc ~ visit, fun.aggregate=length, value.var = "scans")
x2
x3 <- x2[,2:ncol(x2),with=FALSE]/rowSums(x2[,2:ncol(x2), with=FALSE])
x3 <- cbind(x2$upc,x3)

# Then figure out the UPCs with the lowest entropy(?) -- the ones with the most predictive value

# find max value for each row
rowMax <- apply(x3[,2:ncol(x3), with=FALSE], 1, max)
x3$V1[which(rowMax>.3)]


