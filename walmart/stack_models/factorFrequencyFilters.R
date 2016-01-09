#Function to relabel low frequency factors
topK <- function(x,k){
  tbl <- tabulate(x)
  names(tbl) <- levels(x)
  x <- as.character(x)
  levelsToKeep <- names(tail(sort(tbl),k))
  x[!(x %in% levelsToKeep)] <- 'Other'
  out <- factor(x, levels=c(levelsToKeep,"Other"))
  out
}

#Function to relabel factors with frequency below n
topFreq <- function(x,n){
  tbl <- tabulate(x)
  names(tbl) <- levels(x)
  x <- as.character(x)
  levelsToKeep <- names(which(tbl >= n))
  x[!(x %in% levelsToKeep)] <- 'Other'
  out <- factor(x, levels=c(levelsToKeep,"Other"))
  out
}
