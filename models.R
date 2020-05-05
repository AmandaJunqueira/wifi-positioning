#HEADER#
#WifiPositioningProject- Only Training Data 

#Loading Libraries
library(plyr)
library(caret)
library(dplyr)
library(lubridate)
library(ggplot2)
library(e1071)
library(rpart)


#read data
trainingData <- read.csv("data/trainingData.csv", sep = ",")

dataValidation <- read.csv("data/validationData.csv", sep = ",")

#no connection = -105
trainingData[trainingData == 100] <- -105

#Eliminating Duplicates
trainingData <- distinct(trainingData) 

# Transforming Variables
factors <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID")
trainingData[factors] <- lapply(trainingData[factors], factor)

numeric <- c("LONGITUDE", "LATITUDE")
trainingData[numeric] <- lapply(trainingData[numeric], as.numeric)

trainingData$TIMESTAMP <- as_datetime(trainingData$TIMESTAMP)


## NZV
remove_cols <- nearZeroVar(
  trainingData,
  freqCut = 1500,
  uniqueCut = 0.1,
  saveMetrics = FALSE,
  names = FALSE,
  foreach = FALSE,
  allowParallel = TRUE
)

trainingNzv <- trainingData %>% 
  select(-remove_cols)


#selecting only numerical values
pcaDf <- trainingNzv %>% 
  select(starts_with("WAP"),
         LONGITUDE,
         LATITUDE)

#CorerelatedPredictors
desCor <- cor(pcaDf)
highCorr <- sum(abs(desCor[upper.tri(desCor)]) > .950)
summary(desCor[upper.tri(desCor)])
highlyCor <- findCorrelation(desCor, cutoff = .95)
pcaDf <- pcaDf[,-highlyCor]
desCor2 <- cor(pcaDf)
summary(desCor2[upper.tri(desCor2)])

#SAMPLING
sample <- pcaDf %>% 
  sample_n(1000) %>% 
  select(starts_with("WAP"), LATITUDE, LONGITUDE)

#Splitting Data
#select 75/100 for training the model
set.seed(300)
indxTrain <- createDataPartition(y = sample$LATITUDE,
                                 p = 0.75,
                                 list = FALSE)
training <- sample[indxTrain,]
testing <- sample[-indxTrain,]

###KNN###

my_knn_model<- train(LATITUDE ~ .,
                          method = "knn",
                          trControl = trainControl(method = "repeatedcv",
                                                   number = 10,
                                                   repeats = 10),
                          data = training,
                          tuneGrid = expand.grid(k = 2))

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 10)

model <- train(LATITUDE ~ ., data = training, method = "knn",
               preProcess = c('zv', 'pca'),
               trControl = ctrl,
               tuneGrid = expand.grid(k = 2))


#another knn model
set.seed(3527)
subjects <- sample(1:20, size = 80, replace = TRUE)
folds <- groupKFold(subjects, k = 15)


my_knn_model_2<- train(LATITUDE ~ .,
                     method = "knn",
                     trControl = trainControl(method = "repeatedcv",
                                              number = 10,
                                              repeats = 10,
                                              index = folds),
                     data = training,
                     tuneGrid = expand.grid(k = 2))

#SVR#
#Fitting the SVR to the dataset
regressor <- svm(formula = LATITUDE ~ ., 
                 data = training,
                 type = "eps-regression",
                 cross = 7)
 
y_pred <- predict(regressor, testing)


set.seed(825)
svmFit <- train(LATITUDE ~ ., data = training, 
                method = "svmRadial", 
                preProcess = "pca",
                trControl = ctrl,
                tuneLength = 8)

#DECISION TREE REGRESSION#
#Fitting Decision Tree 
regressor_decision <- rpart(formula = LATITUDE ~.,
                            data = training)
y_pred <- predict(regressor_decision, testing)

#Bayesian Regularized Neural Networkds
set.seed(805)
bayesianFit <- train(LATITUDE ~ ., data = training, 
                method = "bayesglm", 
                trControl = ctrl,
                tuneLength = 8)
###POLYNOMINAL REGRESSION###

#SAMPLING
samplePN <- train_set %>% 
  sample_n(1000) %>% 
  select(starts_with("WAP"), LATITUDE)


sampleTest <- test_set %>% 
  sample_n(1000) %>% 
  select(starts_with("WAP"), LATITUDE)

# sampleTest <- test_set 

reg <- lm(formula = LATITUDE ~.,
          data = samplePN)

ggplot() +
  geom_point(aes(x = samplePN$WAP001, 
                 y = samplePN$LATITUDE)) +
  geom_line(aes(x = samplePN$WAP001, 
                y = predict(reg, newdata = samplePN)))





