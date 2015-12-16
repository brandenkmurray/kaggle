library(weatherData)
library(stringr)
library(dplyr)
germStat <- getStationCode("Germany")

# Dictionary for the station data
# !   CD = 2 letter state (province) abbreviation
# !   STATION = 16 character station long name
# !   ICAO = 4-character international id
# !   IATA = 3-character (FAA) id
# !   SYNOP = 5-digit international synoptic number
# !   LAT = Latitude (degrees minutes)
# !   LON = Longitude (degree minutes)
# !   ELEV = Station elevation (meters)
# !   M = METAR reporting station.   Also Z=obsolete? site
# !   N = NEXRAD (WSR-88D) Radar site
# !   V = Aviation-specific flag (V=AIRMET/SIGMET end point, A=ARTCC T=TAF U=T+V)
# !   U = Upper air (rawinsonde=X) or Wind Profiler (W) site
# !   A = Auto (A=ASOS, W=AWOS, M=Meso, H=Human, G=Augmented) (H/G not yet impl.)
# !   C = Office type F=WFO/R=RFC/C=NCEP Center
# !   Digit that follows is a priority for plotting (0=highest)
# !   Country code (2-char) is last column

weatherColnames <- c("CD","STATION","ICAO","IATA","SYNOP","LAT","LONG","ELEV","METAR","NEXRAD","V","UPPER_AIR","AUTO","OFFICE_TYPE","DIGIT","COUNTRY")
stationID <- substr(germStat, 29,32)
city <- str_trim(substr(germStat, 12,28))

stationFrame <- data.frame(city=city, stationID=stationID)
stationFrame <- stationFrame[!(stationFrame$stationID %in% c("","    ")),]

stationFrame$avail <- NULL
#Remove "EDVE" -- gives an error
# stationFrame <- stationFrame[stationFrame$stationID != "EDVE",]
for (i in 1:nrow(stationFrame)){
  avail <- checkSummarizedDataAvailability(station_id = stationFrame$stationID[i],
                                            start_date = "2014-01-01",
                                            end_date="2015-09-17")
  stationFrame[i,"avail"] <- avail
}

stationFrame <- stationFrame[stationFrame$avail==1,]
stationFrame$stationID <- factor(stationFrame$stationID)
weatherDates <- data.frame(start_dates=c("2013-01-01","2014-01-01","2015-01-01"), end_dates=c("2013-12-31","2014-12-31","2015-09-17"), stringsAsFactors=FALSE)

bindWeatherFrame <- NULL
for (i in 1:nrow(stationFrame)){
  tmp <- NULL
  for (j in 1:nrow(weatherDates)) {
    weather <- getSummarizedWeather(station_id = stationFrame$stationID[i], start_date=weatherDates$start_dates[j], end_date=weatherDates$end_dates[j], opt_all_columns=TRUE)
    colnames(weather) <- c("Date", "CEST", "Max_TemperatureF",        
                           "Mean_TemperatureF","Min_TemperatureF", "Max_Dew_PointF",           
                           "MeanDew_PointF", "Min_DewpointF", "Max_Humidity",             
                           "Mean_Humidity", "Min_Humidity", "Max_Sea_Level_PressureIn", "Mean_Sea_Level_PressureIn", "Min_Sea_Level_PressureIn", "Max_VisibilityMiles",      
                           "Mean_VisibilityMiles", "Min_VisibilityMiles", "Max_Wind_SpeedMPH",        
                           "Mean_Wind_SpeedMPH", "Max_Gust_SpeedMPH", "PrecipitationIn",          
                           "CloudCover", "Events", "WindDirDegrees")
    tmp <- rbind(tmp, weather)
    }
  newWeatherFrame <- cbind(city=stationFrame$city[i], stationID=stationFrame$stationID[i],  unname(tmp))
  bindWeatherFrame <- rbind(bindWeatherFrame, newWeatherFrame)  
}

colnames(bindWeatherFrame) <- c("city","stationID",   "Date", "CEST", "Max_TemperatureF",        
  "Mean_TemperatureF","Min_TemperatureF", "Max_Dew_PointF",           
  "MeanDew_PointF", "Min_DewpointF", "Max_Humidity",             
  "Mean_Humidity", "Min_Humidity", "Max_Sea_Level_PressureIn", "Mean_Sea_Level_PressureIn", "Min_Sea_Level_PressureIn", "Max_VisibilityMiles",      
  "Mean_VisibilityMiles", "Min_VisibilityMiles", "Max_Wind_SpeedMPH",        
  "Mean_Wind_SpeedMPH", "Max_Gust_SpeedMPH", "PrecipitationIn",          
  "CloudCover", "Events", "WindDirDegrees")  

write.csv(bindWeatherFrame, "/home/branden/Documents/kaggle/rossman/germanyWeather.csv", row.names=FALSE)
