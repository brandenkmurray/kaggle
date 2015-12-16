library(lubridate)
library(sqldf)
library(data.table)
library(bit64)
library(digest)
library(caret)
library(zoo)
library(plyr)
library(randomForest)
library(xgboost)
library(RPushbullet)
setwd("/home/branden/Documents/kaggle/springleaf")
source("final__utils.R")
# t1 <- fread("train.csv", stringsAsFactors=TRUE, na.strings=c("","-1","97","98","99","999999999","999999998","999","9996","9994","9999","999999996","996","9998","999999995","9995","-99999","999999997","997","994"))
# s1 <- fread("test.csv", stringsAsFactors=TRUE, na.strings=c("","-1","97","98","99","999999999","999999998","999","9996","9994","9999","999999996","996","9998","999999995","9995","-99999","999999997","997","994"))
set.seed(999)
t1 <- fread("train.csv", stringsAsFactors=TRUE)
s1 <- fread("test.csv", stringsAsFactors=TRUE)


s1$target <- 0
t1$filter <- 0
s1$filter <- 2
trainId <- t1$ID
testId <- s1$ID

t1 <- as.data.frame(t1)
s1 <- as.data.frame(s1)
# Change [] to "false" so 0044 gets recognized as a duplicate column
t1[t1=="[]"] <- "false"
dupCols <- names(t1[,duplicated(lapply(t1, digest))])

# Combine the datasets
ts1 <- rbind(t1, s1)
ts1 <- ts1[, !names(ts1) %in% dupCols]

charCols <- names(ts1[sapply(ts1, is.character)])
ts1[charCols] <- lapply(ts1[charCols], function(x) sub("^$","blank",x))
ts1[,charCols] <- lapply(ts1[,charCols], factor)

# Create a filter to seperate train from test
ts1$prob0 <- mean(ts1$target[ts1$filter==0])
ts1$dummy <- as.factor('A')
ts1$y <- ts1$target
ts1$y <- as.factor(paste0("X",ts1$y))

nums <- sapply(ts1, is.numeric)
ts1[nums][is.na(ts1[nums])] <- -999
# zeroVarPreds <- names(ts1[,nearZeroVar(ts1[ts1$filter==0,nums], freqCut=999999/1)])
# ts1 <- ts1[, !names(ts1) %in% zeroVarPreds]

# "VAR_0533","VAR_0541","VAR_0542","VAR_0543","VAR_0544","VAR_0545","VAR_0546","VAR_0547","VAR_0548","VAR_0549","VAR_0551","VAR_0554","VAR_0555","VAR_0556","VAR_0557","VAR_0558","VAR_0561","VAR_0570","VAR_0573","VAR_0574","VAR_0575","VAR_0576","VAR_0577","VAR_0578","VAR_0579","VAR_0583","VAR_0584","VAR_0585","VAR_0586","VAR_0589","VAR_0590","VAR_0591","VAR_0592","VAR_0593","VAR_0596","VAR_0597","VAR_0598","VAR_0599","VAR_0600","VAR_0603","VAR_0606","VAR_0607","VAR_0608","VAR_0609","VAR_0610","VAR_0615","VAR_0616","VAR_0632","VAR_0633","VAR_0634","VAR_0635","VAR_0636","VAR_0637","VAR_0638","VAR_0639","VAR_0640","VAR_0641","VAR_0642","VAR_0643","VAR_0647","VAR_0648","VAR_0649","VAR_0650","VAR_0651","VAR_0652","VAR_0653","VAR_0654","VAR_0655","VAR_0657","VAR_0658","VAR_0659","VAR_0660","VAR_0661","VAR_0662","VAR_0663","VAR_0664","VAR_0665","VAR_0666","VAR_0667","VAR_0668","VAR_0669","VAR_0670","VAR_0671","VAR_0672","VAR_0673","VAR_0674","VAR_0675","VAR_0676","VAR_0677","VAR_0678","VAR_0679","VAR_0681","VAR_0682","VAR_0683","VAR_0684","VAR_0689","VAR_0691","VAR_0694","VAR_0695","VAR_0698","VAR_0699","VAR_0700","VAR_0701","VAR_0703","VAR_0705","VAR_0707","VAR_0708","VAR_0709","VAR_0710","VAR_0711","VAR_0712","VAR_0713","VAR_0714","VAR_0715","VAR_0720","VAR_0721","VAR_0724","VAR_0725","VAR_0727","VAR_0728","VAR_0729","VAR_0731","VAR_0732","VAR_0733","VAR_0734","VAR_0735","VAR_0736","VAR_0737","VAR_0738","VAR_0739","VAR_0743","VAR_0744","VAR_0745","VAR_0746","VAR_0747","VAR_0748","VAR_0749","VAR_0750","VAR_0751","VAR_0752","VAR_0753","VAR_0754","VAR_0755","VAR_0756","VAR_0757","VAR_0759","VAR_0760","VAR_0761","VAR_0762","VAR_0763","VAR_0764","VAR_0765","VAR_0766","VAR_0767","VAR_0769","VAR_0770","VAR_0771","VAR_0772","VAR_0773","VAR_0774","VAR_0775","VAR_0776","VAR_0777","VAR_0779","VAR_0780","VAR_0781","VAR_0782","VAR_0783","VAR_0784","VAR_0785","VAR_0786","VAR_0787","VAR_0788","VAR_0789","VAR_0790","VAR_0791","VAR_0796","VAR_0797","VAR_0798","VAR_0799","VAR_0800","VAR_0801","VAR_0802","VAR_0803","VAR_0804","VAR_0805","VAR_0806","VAR_0807","VAR_0808","VAR_0809","VAR_0810","VAR_0811","VAR_0817","VAR_0837","VAR_0838","VAR_0839","VAR_0841","VAR_0842","VAR_0843","VAR_0844","VAR_0845","VAR_0846","VAR_0848","VAR_0849","VAR_0850","VAR_0851","VAR_0852","VAR_0853","VAR_0854","VAR_0855","VAR_0856","VAR_0857","VAR_0858","VAR_0859","VAR_0860","VAR_0861","VAR_0862","VAR_0863","VAR_0864","VAR_0865","VAR_0866","VAR_0867","VAR_0868","VAR_0869","VAR_0870","VAR_0871","VAR_0872","VAR_0873","VAR_0874","VAR_0875","VAR_0876","VAR_0878","VAR_0879","VAR_0880","VAR_0881","VAR_0882","VAR_0883","VAR_0884","VAR_0885","VAR_0888","VAR_0889","VAR_0890","VAR_0893","VAR_0894","VAR_0897","VAR_0901","VAR_0903","VAR_0904","VAR_0909","VAR_0910","VAR_0912","VAR_0913","VAR_0917","VAR_0918","VAR_0919","VAR_0920","VAR_0921","VAR_0927","VAR_0928","VAR_0929","VAR_0931","VAR_0932","VAR_0933","VAR_0934","VAR_0936","VAR_0937","VAR_0938","VAR_0939","VAR_0941","VAR_0942","VAR_0943","VAR_0944","VAR_0946","VAR_0947","VAR_0948","VAR_0949","VAR_0950","VAR_0951","VAR_0953","VAR_0955","VAR_0956","VAR_0958","VAR_0960","VAR_0961","VAR_0964","VAR_0965","VAR_0966","VAR_0967","VAR_0968","VAR_0970","VAR_0971","VAR_0972","VAR_0976","VAR_0977","VAR_0978","VAR_0979","VAR_0980","VAR_0981","VAR_0982","VAR_0986","VAR_0994","VAR_0995","VAR_1010","VAR_1012","VAR_1013","VAR_1024","VAR_1025","VAR_1041","VAR_1050","VAR_1051","VAR_1055","VAR_1057","VAR_1058","VAR_1059","VAR_1061","VAR_1062","VAR_1063","VAR_1065","VAR_1074","VAR_1075","VAR_1076","VAR_1077","VAR_1081","VAR_1082","VAR_1083","VAR_1084","VAR_1085","VAR_1086","VAR_1087","VAR_1088","VAR_1089","VAR_1090","VAR_1091","VAR_1092","VAR_1093","VAR_1094","VAR_1095","VAR_1096","VAR_1097","VAR_1098","VAR_1099","VAR_1100","VAR_1101","VAR_1102","VAR_1103","VAR_1104","VAR_1105","VAR_1106","VAR_1107","VAR_1108","VAR_1109","VAR_1115","VAR_1116","VAR_1117","VAR_1118","VAR_1119","VAR_1120","VAR_1121","VAR_1122","VAR_1123","VAR_1124","VAR_1125","VAR_1126","VAR_1127","VAR_1128","VAR_1129","VAR_1130","VAR_1131","VAR_1132","VAR_1133","VAR_1134","VAR_1135","VAR_1136","VAR_1137","VAR_1138","VAR_1139","VAR_1140","VAR_1141","VAR_1142","VAR_1143","VAR_1145","VAR_1148","VAR_1149","VAR_1150","VAR_1151","VAR_1152","VAR_1153","VAR_1154","VAR_1155","VAR_1158","VAR_1169","VAR_1170","VAR_1171","VAR_1172","VAR_1173","VAR_1175","VAR_1176","VAR_1179","VAR_1180","VAR_1181","VAR_1182","VAR_1183","VAR_1184","VAR_1186","VAR_1187","VAR_1188","VAR_1189","VAR_1190","VAR_1193","VAR_1194","VAR_1195","VAR_1196","VAR_1197","VAR_1198","VAR_1199","VAR_1200","VAR_1201","VAR_1202","VAR_1203","VAR_1204","VAR_1205","VAR_1206","VAR_1207","VAR_1208","VAR_1209","VAR_1211","VAR_1213","VAR_1214","VAR_1215","VAR_1216","VAR_1218","VAR_1219","VAR_1220","VAR_1221","VAR_1223","VAR_1225","VAR_1227","VAR_1228","VAR_1230","VAR_1232","VAR_1239","VAR_1241","VAR_1242","VAR_1243","VAR_1244","VAR_1245","VAR_1246","VAR_1247","VAR_1248","VAR_1249","VAR_1250","VAR_1251","VAR_1252","VAR_1253","VAR_1254","VAR_1255","VAR_1256","VAR_1257","VAR_1258","VAR_1259","VAR_1260","VAR_1261","VAR_1262","VAR_1263","VAR_1264","VAR_1265","VAR_1266","VAR_1270","VAR_1271","VAR_1272","VAR_1291","VAR_1292","VAR_1301","VAR_1303","VAR_1305","VAR_1306","VAR_1308","VAR_1309","VAR_1310","VAR_1312","VAR_1313","VAR_1314","VAR_1315","VAR_1316","VAR_1317","VAR_1318","VAR_1319","VAR_1320","VAR_1321","VAR_1322","VAR_1323","VAR_1324","VAR_1325","VAR_1326","VAR_1333","VAR_1334","VAR_1335","VAR_1336","VAR_1337","VAR_1338","VAR_1340","VAR_1341","VAR_1342","VAR_1343","VAR_1344","VAR_1346","VAR_1347","VAR_1351","VAR_1353","VAR_1354","VAR_1355","VAR_1356","VAR_1357","VAR_1371","VAR_1372","VAR_1373","VAR_1374","VAR_1375","VAR_1376","VAR_1377","VAR_1378","VAR_1379","VAR_1381","VAR_1382","VAR_1383","VAR_1384","VAR_1385","VAR_1391","VAR_1393","VAR_1394","VAR_1396","VAR_1397","VAR_1398","VAR_1399","VAR_1400","VAR_1401","VAR_1402","VAR_1403","VAR_1405","VAR_1406","VAR_1407","VAR_1408","VAR_1411","VAR_1412","VAR_1413","VAR_1414","VAR_1415","VAR_1416","VAR_1417","VAR_1418","VAR_1419","VAR_1420","VAR_1421","VAR_1422","VAR_1423","VAR_1424","VAR_1425","VAR_1426","VAR_1430","VAR_1431","VAR_1432","VAR_1433","VAR_1434","VAR_1435","VAR_1436","VAR_1437","VAR_1438","VAR_1439","VAR_1440","VAR_1441","VAR_1442","VAR_1443","VAR_1446","VAR_1447","VAR_1448","VAR_1450","VAR_1451","VAR_1452","VAR_1453","VAR_1454","VAR_1458","VAR_1459","VAR_1460","VAR_1466","VAR_1467","VAR_1483","VAR_1484","VAR_1485","VAR_1486","VAR_1489","VAR_1490","VAR_1491","VAR_1492","VAR_1493","VAR_1494","VAR_1495","VAR_1496","VAR_1497","VAR_1498","VAR_1499","VAR_1500","VAR_1501","VAR_1502","VAR_1503","VAR_1504","VAR_1505","VAR_1506","VAR_1507","VAR_1508","VAR_1509","VAR_1510","VAR_1511","VAR_1513","VAR_1514","VAR_1515","VAR_1516","VAR_1517","VAR_1518","VAR_1519","VAR_1520","VAR_1521","VAR_1522","VAR_1523","VAR_1524","VAR_1525","VAR_1526","VAR_1527","VAR_1528","VAR_1529","VAR_1531","VAR_1536","VAR_1541","VAR_1550","VAR_1555","VAR_1560","VAR_1573","VAR_1581","VAR_1582","VAR_1584","VAR_1585","VAR_1586","VAR_1587","VAR_1588","VAR_1590","VAR_1591","VAR_1592","VAR_1593","VAR_1594","VAR_1595","VAR_1596","VAR_1597","VAR_1599","VAR_1600","VAR_1601","VAR_1603","VAR_1604","VAR_1605","VAR_1607","VAR_1608","VAR_1609","VAR_1611","VAR_1612","VAR_1613","VAR_1614","VAR_1615","VAR_1616","VAR_1617","VAR_1618","VAR_1619","VAR_1620","VAR_1621","VAR_1622","VAR_1623","VAR_1624","VAR_1625","VAR_1626","VAR_1627","VAR_1628","VAR_1629","VAR_1630","VAR_1631","VAR_1632","VAR_1633","VAR_1634","VAR_1635","VAR_1636","VAR_1637","VAR_1638","VAR_1640","VAR_1644","VAR_1645","VAR_1646","VAR_1647","VAR_1648","VAR_1649","VAR_1650","VAR_1651","VAR_1652","VAR_1653","VAR_1654","VAR_1655","VAR_1656","VAR_1657","VAR_1658","VAR_1660","VAR_1662","VAR_1663","VAR_1666","VAR_1668","VAR_1669","VAR_1670","VAR_1671","VAR_1672","VAR_1673","VAR_1674","VAR_1675","VAR_1676","VAR_1677","VAR_1678","VAR_1679","VAR_1680","VAR_1681","VAR_1682","VAR_1683","VAR_1684","VAR_1685","VAR_1686","VAR_1687","VAR_1688","VAR_1689","VAR_1690","VAR_1691","VAR_1692","VAR_1693","VAR_1694","VAR_1695","VAR_1696","VAR_1697","VAR_1698","VAR_1699","VAR_1700","VAR_1701","VAR_1702","VAR_1703","VAR_1704","VAR_1705","VAR_1706","VAR_1707","VAR_1708","VAR_1710","VAR_1711","VAR_1712","VAR_1713","VAR_1714","VAR_1715","VAR_1716","VAR_1717","VAR_1718","VAR_1719","VAR_1720","VAR_1721","VAR_1722","VAR_1723","VAR_1724","VAR_1725","VAR_1726","VAR_1727","VAR_1728","VAR_1729","VAR_1730","VAR_1731","VAR_1732","VAR_1733","VAR_1734","VAR_1735","VAR_1736","VAR_1737","VAR_1738","VAR_1739","VAR_1741","VAR_1743","VAR_1744","VAR_1745","VAR_1746","VAR_1747","VAR_1748","VAR_1749","VAR_1750","VAR_1751","VAR_1752","VAR_1753","VAR_1754","VAR_1755","VAR_1756","VAR_1757","VAR_1758","VAR_1759","VAR_1770","VAR_1771","VAR_1795","VAR_1796","VAR_1797","VAR_1798","VAR_1799","VAR_1800","VAR_1801","VAR_1802","VAR_1803","VAR_1804","VAR_1805","VAR_1806","VAR_1807","VAR_1808","VAR_1809","VAR_1810","VAR_1811","VAR_1812","VAR_1813","VAR_1814","VAR_1815","VAR_1816","VAR_1817","VAR_1818","VAR_1819","VAR_1820","VAR_1821","VAR_1822","VAR_1825","VAR_1826","VAR_1827","VAR_1828","VAR_1829","VAR_1830","VAR_1831","VAR_1832","VAR_1833","VAR_1834","VAR_1835","VAR_1836","VAR_1837","VAR_1838","VAR_1839","VAR_1840","VAR_1841","VAR_1842","VAR_1843","VAR_1844","VAR_1845","VAR_1846","VAR_1847","VAR_1848","VAR_1849","VAR_1850","VAR_1851","VAR_1852","VAR_1855","VAR_1858","VAR_1859","VAR_1860","VAR_1861","VAR_1862","VAR_1863","VAR_1864","VAR_1865","VAR_1867","VAR_1868","VAR_1869","VAR_1870","VAR_1871","VAR_1874","VAR_1885","VAR_1888","VAR_1889","VAR_1890","VAR_1891","VAR_1892","VAR_1893","VAR_1894","VAR_1895","VAR_1896","VAR_1897","VAR_1900","VAR_1901","VAR_1902","VAR_1905","VAR_1912","VAR_1913","VAR_1914","VAR_1915","VAR_1916","VAR_1917","VAR_1918","VAR_1919","VAR_1922","VAR_1923","VAR_1924","VAR_1925","VAR_1926","VAR_1927","VAR_1928","VAR_1929","VAR_1930","VAR_1931","VAR_1932","VAR_1933",
# 
# 
# fact <- sapply(ts1, is.factor)
# 
# vars <- grepl("VAR", names(ts1))
# ts1[vars] <- lapply(ts1[vars], function(x) replace(x, x %in% c("",-1,97,98,99,999999999,999999998,999,9996,9994,9999,999999996,996,9998,999999995,9995,-99999,999999997,997,998,994),NA))
## DATE FACTORS


# ts1 <- fread("ts1Save1.csv", stringsAsFactors=TRUE)
# ts1 <- as.data.frame(ts1, stringsAsFactors=TRUE)
# charCols <- names(ts1[sapply(ts1, is.character)])
# ts1[,charCols] <- lapply(ts1[,charCols], factor)

# 0207 is all NA, 212 is redundant()
ts1$VAR_0207 <- NULL
ts1$VAR_0212 <- NULL

ts1$VAR_0241_4 <- as.factor(substr(ts1$VAR_0241, 1, 4))
ts1$VAR_0241_3 <- as.factor(substr(ts1$VAR_0241, 1, 3))
# ddply(ts1[ts1$filter==0,], .(VAR_0166_wday), summarise, avg=mean(target), n=length(target))

# ts1 <- ts1[, !names(ts1) %in% dates]
################ NEED TO FIGURE OUT HOW TO DEAL WITH NAs

# naVals <- c(97,98,99,999999999,999999998,999,998,9996,9994,9999,999999996,996,9998,999999995,9995,-99999,999999997,997,994)
## TRY TO PARALLELIZE!!!!
# for (i in naVals) {
#   ts1[nums][ts1[nums] == i] <- -1
# }


# zip <- substr(t1$VAR_0212, 1,5)
# city <- t1$VAR_0200
# state <- t1$VAR_0274
# vars <- grepl("VAR", names(ts1))
# varnames <- names(ts1[vars])
# fact <- sapply(ts1, is.factor)
# nums <- sapply(ts1, is.numeric)
# numnames <- names(ts1[nums])
# dates <- c("VAR_0073","VAR_0075","VAR_0156", "VAR_0157","VAR_0158", "VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
# ts1[dates] <- lapply(ts1[dates], dmy_hms)
# ts1$VAR_daydiff_204_217 <- ts1$VAR_0204 - ts1$VAR_0217

set.seed(600)
# Factors w/ more than 10 levels "VAR_0200","VAR_0214","VAR_0237","VAR_0274","VAR_0325","VAR_0342","VAR_0404","VAR_0493"
highCardVars <- c("VAR_0200","VAR_0214","VAR_0237","VAR_0241","VAR_0241_4", "VAR_0241_3", "VAR_0274","VAR_0325","VAR_0342","VAR_0404","VAR_0493")
for(ii in highCardVars) {
  print(ii)
  x <- as.numeric(ts1[, ii])
  ts1[, paste(ii, "_num", sep="")] <- x
}

## One way count
for(ii in highCardVars) {
  print(ii)
  x <- data.frame(x1=ts1[, ii])
  sum1 <- sqldf("select x1, sum(1) as cnt
                from x  group by 1 ")
  tmp <- sqldf("select cnt from x a left join sum1 b on a.x1=b.x1")
  ts1[, paste(ii, "_cnt", sep="")] <- tmp$cnt
}




## Date dimensions
dates <- c("VAR_0073","VAR_0075","VAR_0156", "VAR_0157","VAR_0158", "VAR_0159","VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177","VAR_0178","VAR_0179","VAR_0204","VAR_0217")
# ts1[dates] <- lapply(ts1[dates], function(x) strptime(x, "%d%h%y:%H:%M:%S"))

## Create numeric date columns
for(ii in dates) {
  print(ii)
  ts1[, paste(ii, "_num", sep="")] <- as.numeric(strptime(ts1[[ii]], "%d%h%y:%H:%M:%S"))
  ts1[, paste(ii,"_wday", sep="")] <- strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")$wday
  ts1[, paste(ii,"_hour", sep="")] <- strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")$hour
  ts1[, paste(ii,"_year", sep="")] <- strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")$year
  ts1[, paste(ii,"_mon", sep="")] <- strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")$mon
  ts1[, paste(ii,"_mday", sep="")] <- strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")$mday
  ts1[, paste(ii,"_yearmon", sep="")] <- as.numeric(as.yearmon(strptime(ts1[[ii]], "%d%h%y:%H:%M:%S")))
}

## Create year-month variable
# for (i in dates){
#   tmp <- factor(ifelse(is.na(as.character(as.yearmon(ts1[,i]))),"NA",as.character(as.yearmon(ts1[,i]))), exclude=NULL)
#   ts1 <- cbind(ts1, tmp)
#   colnames(ts1)[ncol(ts1)] <- paste0(i,"_yearmon")
# }

## Target averages for year-month variable
# for (i in paste0(dates,"_yearmon")){
#   tmp <- cat2WayAvg(ts1, i,"dummy","target","prob0",filter=ts1$filter==0, k=20, f=10, r_k=0.05)
#   ts1 <- cbind(ts1, tmp)
#   colnames(ts1)[ncol(ts1)] <- paste0("exp_",i)
# }

## Create weekday variable
# for (i in dates){
#   tmp <- factor(ifelse(is.na(as.character(wday(ts1[,i]))),"NA",as.character(wday(ts1[,i]))), exclude=NULL)
#   ts1 <- cbind(ts1, tmp)
#   colnames(ts1)[ncol(ts1)] <- paste0(i,"_wday")
# }

## Target averages for wday variable



dateCombos <- combn(c(dates), 2, simplify=FALSE)
for (i in dateCombos) {
  days <- abs(as.numeric(difftime(time1 = strptime(ts1[,unlist(i)[1]], "%d%h%y:%H:%M:%S"), time2 = strptime(ts1[,unlist(i)[2]], "%d%h%y:%H:%M:%S"), units="days")))
  days[is.na(days)] <- -999
  ts1 <- cbind(ts1, days)
  colnames(ts1)[ncol(ts1)] <- paste0("daydiff_",unlist(i)[1], "_",unlist(i)[2])
}

# Drop original high cardinality and date variables from the data set



ts1[ts1==-999] <- NA
ts1 <- na.roughfix(ts1)

set.seed(500)
# Moving exps to here because cat2WayAvg doesn't handle NAs well
for (i in outer(dates, c("_hour","_mday","_yearmon","_mon","_wday","_year","_num"), paste0)){
  tmp <- cat2WayAvg(ts1, i,"dummy","target","prob0",filter=ts1$filter==0, k=20, f=10, r_k=0.05)
  ts1 <- cbind(ts1, tmp)
  colnames(ts1)[ncol(ts1)] <- paste0("exp_",i)
}

for(i in highCardVars) {
  print(i)
  tmp <- cat2WayAvg(ts1, i,"dummy","target","prob0",filter=ts1$filter==0, k=20, f=10, r_k=0.1)
  ts1 <- cbind(ts1, tmp)
  colnames(ts1)[ncol(ts1)] <- paste0("exp_",i)
}

highCardCombos <- combn(c(highCardVars), 2, simplify=FALSE)
for(i in highCardCombos) {
  print(i)
  tmp <- cat2WayAvg(ts1, unlist(i)[1], unlist(i)[2],"target","prob0",filter=ts1$filter==0, k=20, f=10, r_k=0.1)
  ts1 <- cbind(ts1, tmp)
  colnames(ts1)[ncol(ts1)] <- paste0("exp_",unlist(i)[1], "_",unlist(i)[2])
}

dropVars <- c(highCardVars, dates)
ts1 <- ts1[, !names(ts1) %in% dropVars]

# zeroVarPreds <- names(ts1[,nearZeroVar(ts1[ts1$filter==0,nums], freqCut=999999/1)])
# ts1 <- ts1[, !names(ts1) %in% zeroVarPreds]
varnames <- names(ts1[,grepl("VAR", names(ts1))])
# trainInd <- data.frame(model.matrix(y ~ ., data=ts1[ts1$filter==0, c("y",varnames)]))[,-1]
# testInd <- data.frame(model.matrix(y ~ ., data=ts1[ts1$filter==2, c("y",varnames)]))[,-1]
# trainInd$y <- ts1$y[ts1$filter==0]
# testInd$y <- ts1$y[ts1$filter==2]

train <- ts1[ts1$filter==0,]
test <- ts1[ts1$filter==2,]
train <- train[!names(train) %in% c("dummy","filter","prob0","y")]
test <- test[!names(test) %in% c("dummy","filter","prob0","y")]

# write.csv(ts1, "ts1-09-21-2015.csv")
# write.csv(testInd, "testInd-09-20-2015.csv", row.names=FALSE)
# rm(ts1)
rm(s1)
rm(t1)
rm(x)
rm(sum1)

gc()



varnames <- names(train[,grepl("VAR", names(train))])
param <- list( eta                 = 0.005,
               max_depth           = 16,  # changed from default of 8
               subsample           = 0.8,
               colsample_bytree    = 0.8
               
)

set.seed(10)
tme <- Sys.time()
xgb10 <- xgboost(data        = data.matrix(train[,varnames]),
                label       = train$target,
                params = param,
                print.every.n = 100,
                nrounds     = 20000,
                objective   = "binary:logistic",
                eval_metric = "auc")
(runTime <- Sys.time() - tme)
pbPost(type = "note", title = "XGB10", body="Done.")
save(xgb10, file="xgb10.rda")

targetPred <- predict(xgb10, data.matrix(test[,varnames]))
submission <- data.frame(ID=test$ID, target=targetPred)
# colnames(submission)[2] <- "target"
write.csv(submission, "submission-09-25-2015-xgb10.csv", row.names=FALSE)
