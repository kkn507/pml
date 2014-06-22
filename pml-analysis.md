# Human Activity Recognition data Analysis
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

### Load data

```r
trainData <- read.csv("pml-training.csv", header=TRUE,na.strings=c("NA","","#DIV/0!"))
testData <- read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA","","#DIV/0!"))
```

### Clear data
We get rid from first seven features cause first we don't need some of them and second: for example num_window answers the question so no application for machine learning. Also many feature are NA in test set? so we find and remove them from both sets


```r
all.na <- sapply(testData, function(x) all(is.na(x)))

trainData <- trainData[,which(!all.na)][-c(1:7)]
testData <- testData[,which(!all.na)][-c(1:7)]

trainData$classe <- as.factor(trainData$classe)
```
## Buid model
We create model using caret package, random forest algorithm and ten-fold cross validation. (time consuming process so cache the result). To estimate out-of-sample error we split training set in two parts (70/30).


```r
set.seed(2334)

library(caret)
inTrain <- createDataPartition(trainData$classe, p=0.70,list=FALSE)
training <- trainData[inTrain,]
testing <- trainData[-inTrain,]

tc <- trainControl(method = "cv", number = 10)

modFit <- train(
  training$classe ~ ., 
  data=training[,-dim(training)[2]], 
  method="rf", 
  trControl = tc,
  prox=TRUE
  )
```

## Make some prediction and estimate out of sample error

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pred <- predict(modFit,testing[,-dim(testing)[2]])
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
cm <- confusionMatrix(pred, testing$classe)
ce.out <- 1 - cm$overall["Accuracy"][[1]]
print(cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    7    0    0    0
##          B    0 1130   10    0    0
##          C    0    2 1012   17    0
##          D    0    0    4  947    0
##          E    1    0    0    0 1082
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.995)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.986    0.982    1.000
## Specificity             0.998    0.998    0.996    0.999    1.000
## Pos Pred Value          0.996    0.991    0.982    0.996    0.999
## Neg Pred Value          1.000    0.998    0.997    0.997    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.161    0.184
## Detection Prevalence    0.285    0.194    0.175    0.162    0.184
## Balanced Accuracy       0.999    0.995    0.991    0.991    1.000
```
### Out-of-sample Error 0.007


