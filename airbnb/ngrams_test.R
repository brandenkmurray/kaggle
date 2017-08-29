library(dplyr)
library(data.table)
library(tm)
library(SnowballC)
library(RWeka)
sess <- data.table(read.csv("./sessions.csv"))

sessTrans <- sess[, list(actionString=toString(action)),
                  by=list(user_id)]

corpus0404 <- Corpus(VectorSource(sessTrans$actionString))
corpus0404 <- tm_map(corpus0404, PlainTextDocument)
corpus0404 <- tm_map(corpus0404, removePunctuation)
freq0404 <- DocumentTermMatrix(corpus0404, control=list(tokenize = NGramTokenizer))
findFreqTerms(freq0404, lowfreq=10)
sparse0404 <- removeSparseTerms(freq0404, .99989)
df0404 <- as.data.frame(as.matrix(sparse0404))

library(dplyr)
library(data.table)
require(quanteda)
sess <- data.table(read.csv("./Documents/kaggle/airbnb/sessions.csv"))
sessTrans <- sess[, list(actionString=toString(action),
                         actionTypeString=toString(action_type),
                         actionTypeDetString=toString(action_detail),
                         deviceString=toString(device_type)),
                  by=list(user_id)]

actionTokens <- tokenize(toLower(sessTrans$actionString), removePunct = TRUE, ngrams = 2)
actionTypeTokens <- tokenize(toLower(sessTrans$actionTypeString), removePunct = TRUE, ngrams = 2)
actionTypeDetTokens <- tokenize(toLower(sessTrans$actionTypeDetString), removePunct = TRUE, ngrams = 2)
deviceTokens <- tokenize(toLower(sessTrans$deviceString), removePunct = TRUE, ngrams = 2) 


actionDFM <- dfm(x = actionTokens)
actionTypeDFM <- dfm(x = actionTypeTokens)
actionTypeDetDFM <- dfm(x = actionTypeDetTokens)
deviceDFM <- dfm(x = deviceTokens)


actionDFM_feats <- names(topfeatures(actionDFM, n=100))
actionTypeDFM_feats <- names(topfeatures(actionTypeDFM, n=100))
actionTypeDetDFM_feats <- names(topfeatures(actionTypeDetDFM, n=100))
deviceDFM_feats <- names(topfeatures(deviceDFM, n=100))

actionDF <- as.data.frame(actionDFM[,actionDFM_feats])
actionTypeDF <- as.data.frame(actionTypeDFM[,actionTypeDFM_feats])
actionTypeDetDF <- as.data.frame(actionTypeDetDFM[,actionTypeDetDFM_feats])
deviceDF <- as.data.frame(deviceDFM[,deviceDFM_feats]) 

binded <- data.frame(user_id=sessTrans$user_id, actionDF, actionTypeDF, actionTypeDetDF, deviceDF)






