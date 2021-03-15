setwd("D:\\Data Science\\edwisor\\Project Stage\\Project2")

data_set <- read.csv("train_cab.csv")

str(data_set)
head(data_set, n = 5)
summary(data_set)

###Typecasting Variables#####

data_set$fare_amount <- as.numeric(as.character(data_set$fare_amount))

library(lubridate)

data_set$pickup_datetime <- as_datetime(as.character(data_set$pickup_datetime))

summary(data_set)

###Splitting datetime variable######
data_set$pickup_year <- (year(data_set$pickup_datetime))
data_set$pickup_month <- (month(data_set$pickup_datetime))
data_set$pickup_day <- (mday(data_set$pickup_datetime))
data_set$pickup_wday <- (wday(data_set$pickup_datetime))
data_set$pickup_hour <- (hour(data_set$pickup_datetime))

###Dropping 'pickup_datetime' Variable####
library(dplyr)
data_set <- data_set %>%
        select(-c('pickup_datetime'))

#######For Cleaning
write.csv(data_set, "data_before_clean.csv", row.names = F)


###Setting irrelevant fare_amount, passenger_count,  latitudes and longitudes values to NA###

library(tidyr)



data_set <- data_set %>%
        mutate(across(contains("latitude"), ~ifelse(.x <= 90 & .x >= -90, .x, NA))) %>%
        mutate(across(contains("longitude"), ~ifelse(.x <= 180 & .x >= -180, .x, NA))) %>%
        mutate(fare_amount = ifelse(fare_amount > 0, fare_amount, NA)) %>%
        mutate(passenger_count = ifelse(passenger_count < 7, passenger_count, NA))

#########Setting decimal passenger count to NA

data_set$passenger_count[data_set$passenger_count%%1!=0] = NA
data_set$passenger_count = factor(data_set$passenger_count)

sum(is.na(data_set))



write.csv(data_set, file = "Data_before_out.csv", row.names = F)

#######OUTLIER ANALYSIS########




###Checking Outliers from Latitudes and longitudes using Maps Visualization from tableau########
##Importing image of Graph
###install.packages("imager")

library(imager)

outlier_coords <- load.image("Outlier_Analysis_coords.png")
plot(outlier_coords)




###Finding upper and lower limits of Latitude and Longitude values#######

Hor_Point1 <- c(-73.7260, 40.6865)
Hor_Point2 <- c(-74.1860, 40.6939)
Ver_Point1 <- c(-73.8842, 40.9560)
Ver_Point2 <- c(-73.8584, 40.5747)

#####Setting Latitude and Longitude Outliers to NA########
data_set <- data_set %>%
        mutate(across(contains("latitude"), ~ifelse(.x <= 40.9560 & .x >= 40.5747, .x, NA))) %>%
        mutate(across(contains("longitude"), ~ifelse(.x <= -73.7260 & .x >= -74.1860, .x, NA)))


na_index <- unique(which(is.na(data_set), arr.ind = TRUE)[, 1])
na_data <- data_set[na_index, ]
        
summary(data_set)

####Feature Engineering#####

###Creating the new variables based on Distance


###Finding max distance between above points####
# install.packages("geosphere")
library(geosphere)



###Creating a new variable####

dfm <- distHaversine(cbind(data_set$pickup_longitude, data_set$pickup_latitude), cbind(data_set$dropoff_longitude, data_set$dropoff_latitude))

data_set <- data_set %>%
        mutate(Distance = dfm)

####Dropping latitude and longitude variables
data_set <- data_set %>%
        select(-c(contains("latitude"), contains("longitude")))


####Box Plot for Outlier Analysis#####
library(ggplot2)

g1 <- ggplot(data = data_set, aes(y = Distance)) + geom_boxplot() 
g2 <- ggplot(data = data_set, aes(y = fare_amount)) + geom_boxplot()




sum(data_set$fare_amount>100, na.rm = TRUE)
data_set <- data_set[data_set$fare_amount < 100 & !is.na(data_set$fare_amount), ]
data_set <- data_set[!is.na(data_set$pickup_hour), ]



###Setting outliers in Distance to NA#####

data_set$Distance[data_set$Distance > 64000] <- NA
data_set$Distance[data_set$Distance <= 0] <- NA

sum(is.na(data_set))
summary(data_set)


###Imputing NA's with knnImputation Value####   

data2 <-data_set

str(data2)

data2$pickup_year <- as.factor(data2$pickup_year)
data2$pickup_month <- as.factor(data2$pickup_month)
data2$pickup_day <- as.factor(data2$pickup_day)
data2$pickup_wday <- as.factor(data2$pickup_wday)
data2$pickup_hour <- as.factor(data2$pickup_hour)


data2[346, 8]  ####3085.068
data2[346, 8] = NA
data2$Distance[is.na(data2$Distance)] = median(data2$Distance, na.rm = TRUE)

library(DMwR)
set.seed(123)
data_knn <- knnImputation(data = data2, k = 5)

sum(is.na(data_knn))

data_set <- data_knn
data2 <- data_knn
rm(data_knn)
rm(na_data)

summary(data_set)

#####Rounding the values in passenger_count#####
# data_set$passenger_count <- round(data_set$passenger_count, digits = 0)
# data_set$passenger_count <- as.factor(data_set$passenger_count)

write.csv(data_set, "data_for_EDA.csv", row.names = F)

str(data_set)


########Feature Engineering based on EDA########

##Assigning session_categories to pickup_hour
data_set$pickup_hour = as.numeric(as.character(data_set$pickup_hour))
summary(data_set$pickup_hour)
data_set$pickup_hour[data_set$pickup_hour <= 5 & data_set$pickup_hour >= 0] = 1
data_set$pickup_hour[data_set$pickup_hour >= 6 & data_set$pickup_hour <= 12] = 2
data_set$pickup_hour[data_set$pickup_hour >= 13 & data_set$pickup_hour <= 16] = 3
data_set$pickup_hour[data_set$pickup_hour >= 17 & data_set$pickup_hour <= 20] = 4
data_set$pickup_hour[data_set$pickup_hour >= 21 & data_set$pickup_hour <= 23] = 5
data_set$pickup_hour = factor(data_set$pickup_hour, labels = c("Midnight", "Morning", "Afternoon", "Evening", "Night"))

##Assigning season_categories to pickup_month  
data_set$pickup_month = as.numeric(as.character(data_set$pickup_month))

data_set$pickup_month[data_set$pickup_month >= 10 | data_set$pickup_month == 1] = 1
data_set$pickup_month[data_set$pickup_month >= 2 & data_set$pickup_month <= 5] = 2
data_set$pickup_month[data_set$pickup_month >= 6 & data_set$pickup_month <= 7] = 3
data_set$pickup_month[data_set$pickup_month >= 8 & data_set$pickup_month <= 10] = 4
summary(data_set$pickup_month)
data_set$pickup_month = factor(data_set$pickup_month, labels = c("Winter", "Spring", "Summer", "Autumn"))
##################################


###########CORRELATION ANALYSIS##########


numeric_data <- data_set %>%
        select_if(is.numeric)

factor_data <- data_set %>%
        select_if(is.factor)

library(corrr)

cor_mat <- cor(numeric_data[, -c(1)])

cor_table <- correlate(numeric_data) %>%
        shave() %>%
        stretch()
cor_table <- cor_table[!is.na(cor_table$r), ]

# install.packages("corrplot")
library(corrplot)

cor_test <- corrplot(cor_mat, method = "color", type = "upper")

# install.packages("usdm")
library(usdm)

vif(numeric_data[, -1])


##########ANOVA TEST#########
factor_data_aov <- data.frame(fare_amount = numeric_data$fare_amount, factor_data)
aov_data <- data.frame(Variable_Name = names(factor_data))
p_value <- {}

for(i in names(factor_data_aov)[-c(1)]){
        aov_test <- aov(fare_amount ~ factor_data[[i]], data = factor_data_aov)
        p_value <- c(p_value, summary(aov_test)[[1]][1, 5])
}

aov_data$p_value <- p_value

aov_data$Result <- aov_data$p_value <= 0.05


factor_data <- factor_data %>%
        select(-c(pickup_day, pickup_wday))



####Feature Scaling##########

library(e1071)
library(scales)
ggplot(data = numeric_data, aes(x = Distance)) + geom_histogram()

numeric_data$Distance = rescale(numeric_data$Distance)
data_scaled <- data.frame(numeric_data, factor_data)


sum(is.na(data_scaled))
summary(data_scaled)


rm(factor_data_aov)

###Converting Non-Ordinal Categorical variable using One-Hot Encoding##
###Creating Dummy VAriables as part of One-Hot Encoding Scheme###

str(data_scaled)

library(caret)

dummy <- dummyVars("~.", data = data_scaled[, -c(3)])
dummy_data <- data.frame(predict(dummy, newdata = data_scaled))
final_data <- data.frame(dummy_data, data_scaled[["passenger_count"]])


set.seed(123456)
train_index = createDataPartition(final_data$fare_amount, p = 0.8, list = FALSE)
train <- final_data[train_index,]
test <- final_data[-train_index,]

###############Supervised Machine Learning##########


###Linear Regression Model##########
set.seed(789)
lr_model <- lm(fare_amount ~ ., data = train)
pred_train <- predict(lr_model, train[, -c(1)])
pred_vd <- predict(lr_model, test[, -c(1)])

LR_error <- postResample(pred_train, obs = train$fare_amount) ###RMSE:4.678, MAE:2.338, R2: 0.751
LR_error_vd <- postResample(pred_vd, obs = test$fare_amount) ####RMSE:4.881, MAE: 2.34, R2: 0.727
print(AIC(lr_model))  ###76010.2

########Random Forest Model########
library(randomForest)
set.seed(678)
rf_model <- randomForest(fare_amount ~ ., data = train, ntree = 500, importance = TRUE)
pred_train_rf <- predict(rf_model, train)
pred_vd_rf <- predict(rf_model, test)

RF_error <- postResample(pred_train_rf, obs = train$fare_amount) ###RMSE:3.304, MAE:1.786, R2:0.884
RF_error_vd <- postResample(pred_vd_rf, obs = test$fare_amount) ####RMSE:4.833, MAE: 2.413, R2:0.734 

####Tuning model using RandomSearchCV########
set.seed(587)
best_rf <- tuneRF(train[, -c(1)], train$fare_amount, stepFactor = 1.2, ntreeTry = 500)
rf_model <- randomForest(fare_amount ~ ., data = train, ntree = 400, importance = TRUE, mtry = 7)
pred_train_rf <- predict(rf_model, train)
pred_vd_rf <- predict(rf_model, test)

RF_error <- postResample(pred_train_rf, obs = train$fare_amount) ###RMSE:3.075, MAE:1.616, R2:0.898
RF_error_vd <- postResample(pred_vd_rf, obs = test$fare_amount) ####RMSE:4.842, MAE: 2.392, R2:0.732 

########XGBoost Model###########
set.seed(4567)
library(xgboost)

dtrain_label <- final_data$fare_amount
dtrain <- xgb.DMatrix(data = data.matrix(final_data[, -c(1)]), label = dtrain_label)
xg_model <- xgboost(dtrain, nrounds = 3000, print_every_n = 100)
        
pred_train_xg <- predict(xg_model, data.matrix(train[, -c(1)]))
pred_vd_xg <- predict(xg_model, data.matrix(test[, -c(1)]))

XG_error <- postResample(pred_train_xg, obs = train$fare_amount)####RMSE:1.694, MAE:0.365, R2:0.967
XG_error_vd <- postResample(pred_vd_xg, obs = test$fare_amount)####RMSE:2.225, MAE:0.369, R2:0.943

