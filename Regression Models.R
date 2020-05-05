###CLASSIFICATION MODELS###

#read data
trainingData <- read.csv("data/trainingData.csv", sep = ",")

dataValidation <- read.csv("data/validationData.csv", sep = ",")

#no connection = -105
trainingData[trainingData == 100] <- -105
dataValidation[dataValidation == 100] <- -105


#Eliminating Duplicates
trainingData <- distinct(trainingData) 
dataValidation <- distinct(dataValidation)

# Transforming Variables
factors <- c("FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID")
trainingData[factors] <- lapply(trainingData[factors], factor)

numeric <- c("LONGITUDE", "LATITUDE")
trainingData[numeric] <- lapply(trainingData[numeric], as.numeric)

trainingData$TIMESTAMP <- as_datetime(trainingData$TIMESTAMP)

#Transforming Variables - Validation
dataValidation[factors] <- lapply(dataValidation[factors], factor)
dataValidation[numeric] <- lapply(dataValidation[numeric], as.numeric)
dataValidation$TIMESTAMP <- as_datetime(dataValidation$TIMESTAMP)



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

testingNzv <- dataValidation %>% 
  select(-remove_cols)

#selecting only numerical values
classificationDf <- trainingNzv %>% 
  select(starts_with("WAP"),
         FLOOR,
         BUILDINGID)

classificationTest <- testingNzv %>% 
  select(starts_with("WAP"),
         FLOOR,
         BUILDINGID)

#SAMPLING
sampleClass <- classificationDf %>% 
  sample_n(1000) %>% 
  select(starts_with("WAP"), FLOOR, BUILDINGID)

sampleClassValid <- classificationTest %>% 
  sample_n(1000) %>% 
  select(starts_with("WAP"), FLOOR, BUILDINGID)

#Splitting Data
#select 75/100 for training the model
set.seed(300)
indxTrain <- createDataPartition(y = sampleClass$BUILDINGID,
                                 p = 0.75,
                                 list = FALSE)
trainingClass <- sampleClass[indxTrain,]
testingClass <- sampleClass[-indxTrain,]

levels(trainingClass$BUILDINGID) <- c("buildingZero", "buildingOne", "buildingTwo")
levels(trainingClass$FLOOR) <- c("groundFloor", "firstFloor", "secondFloor", "thirdFloor", "forthFloor")

levels(testingClass$BUILDINGID) <- c("buildingZero", "buildingOne", "buildingTwo")
levels(testingClass$FLOOR) <- c("groundFloor", "firstFloor", "secondFloor", "thirdFloor", "forthFloor")

levels(sampleClassValid$BUILDINGID) <- c("buildingZero", "buildingOne", "buildingTwo")
levels(sampleClassValid$FLOOR) <- c("groundFloor", "firstFloor", "secondFloor", "thirdFloor", "forthFloor")


#RDA, #ROC
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           classProbs = TRUE,
                           search = "random")

set.seed(825)
rda_fit <- train(BUILDINGID ~ ., data = trainingClass, 
                 method = "rda",
                 metric = "ROC",
                 trControl = fitControl)
rda_fit


#BAGGED TREE
baggedFit <- train(BUILDINGID ~ ., data = trainingClass,
                   method = "treebag",
                   nbagg = 50,
                   metric = "ROC",
                   trControl = fitControl)


#SVM
svmFit <- train(BUILDINGID ~ ., data = trainingClass,
                method = "svmRadial", 
                trControl = fitControl, 
                preProc = c("nzv", "center", "scale"),
                metric = "ROC",
                tuneLength = 15)

regressionPredict <- predict(baggedFit, testingClass)
postResample(pred = regressionPredict, obs = sampleClassTest)
confusionMatrix(data = regressionPredict, reference = testingClass$BUILDINGID)

predictionValid <- predict(baggedFit, sampleClassValid)
confusionMatrix(data = predictionValid, reference = sampleClassValid$BUILDINGID)












