# Activity Prediction-Machine Learning Project
Jess Holling  
Feburary 6, 2016  

# Project Summary
From the research project "Qualitative Activity Recognition of Weight Lifting Exercizes" by Groupware@LES, six males performed specific Weight Lift Exercises. These males were fitted with four sensors (belt, arm, glove, dumbell) to record exercise movements covering a 0.5 to 2.5 second window. Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 

 * (Class A)- No mistakes & exactly according to the specification
 * (Class B)- Mistake of throwing the elbows to the front 
 * (Class C)- Mistake of lifting the dumbbell only halfway   
 * (Class D)- Mistake of lowering the dumbbell only halfway 
 * (Class E)- Mistake of throwing the hips to the front 

The goal of my project is to take the same weight lifting database and develop a highly accurate prediction model that can determine, via the "classe" variable, how well the exercise was completed (i.e proper-A or which mistake-B/C/D/E).

I will employ two machine learning models: Random Forest & Support Vector Machines (SVM). These models are efficient for large variables & classification analysis. The highest accuracy model on a validating data set will be choosen as "best" model.

This "best" new model will then be applied to the "test" data set consisting of 20 observations without the "classe" a.k.a. exercise quality known and entered to the Course Project Prediction Quiz and only the code shown below.

# Results
The fitted model for Random Forest gave 99.08% accuracy while SVM gave 99.45% accuracy based on the validation data. Both models are great "fits" exceeing 99% accuracy. The SVM is chosen for having 0.37% less error rate and higher Kappa then Random Forest and will be used for the test data set. 

### Load data and clean data

```r
set.seed(100)
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(doParallel))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(kernlab))
suppressPackageStartupMessages(library(corrplot))

if(!file.exists("pml-training.csv")) #checks if file is download to working directory
{  download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                 "pml-training.csv")
   download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                 "pml-testing.csv")}

train <- read.csv("pml-training.csv", na.strings = c("NA", ""))
test <- read.csv("pml-testing.csv") # loading 20 test observations
dim(train)
```

```
## [1] 19622   160
```

```r
str(train[1:6,1:12]) #sample of 160 columns with 12 rows
```

```
## 'data.frame':	6 obs. of  12 variables:
##  $ X                   : int  1 2 3 4 5 6
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232
##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1
##  $ num_window          : int  11 11 11 12 12 12
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45
##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4
##  $ total_accel_belt    : int  3 3 3 3 3 3
##  $ kurtosis_roll_belt  : Factor w/ 396 levels "-0.016850","-0.021024",..: NA NA NA NA NA NA
```

```r
#Removing columns that contain NA Note: these cols also have some calculations too#
noNA<-sapply(train,function(x) sum(is.na(x)))==0
train.noNA <- train[,noNA]
test.noNA<-test[,noNA]

#Remove other columns that will not be used 
train.tidy = train.noNA[,!grepl("^X|raw|cvtd|new|num",names(train.noNA))]
test.tidy =  test.noNA[,!grepl("^X|raw|cvtd|new|num|prob",names(test.noNA))]

train.split<-createDataPartition(y=train.tidy$classe,p=0.75, list=F)
training<-train.tidy[train.split,]   #splitting data set to 75/25 of original
validate<-train.tidy[-train.split,]    #will use to verify model is good fit
```
Looking at the given "train" data set (19622x160), several columns containing "NA"s also have calculations (i.e "var", "mean", etc) at the end of each short window section.  I choose to remove those as well as other non needed columns. 

The variables were adjusted from 160 down to 54. Lastly, this data set was split 75% into "train.tidy" & 25% into "validate". 

### Correlation plot

```r
cor.train <- cor(training[, -c(1,54)])
corrplot(cor.train, method=c("color"), bg = "white", addgrid.col = "gray50", tl.cex=.75,
         type="lower", l.col = "black", col = colorRampPalette(c("red","white","blue"))(100))
```

![](Excersize_Activity1_files/figure-html/correlation plot-1.png)<!-- -->

A few of the variables (i.e. yaw_belt & roll_belt) look to be highly correlated (more color = more correlation) and need to be considered in models.

## Fit new models
Adding 10-k fold cross validation to models will test multiple variables across multiple resamples and choose the best overall. This will further increase accuracy and lower overfit. The downside is processing time. So parallel processing has been added for multiply processor cores.


```r
cluster <- makeCluster(detectCores() - 1) #use all but 1 processor for calcs
registerDoParallel(cluster)

#setting for 10 K-fold cross validation & mulit processors
fitControl <- trainControl(method ="cv",number=10, allowParallel = TRUE) 
```

### Random Forest model

```r
fit.rf<-train(classe~.,method="rf" ,data=training, trControl=fitControl) 
confusionMatrix(fit.rf) 
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.1 17.3  0.3  0.0
##          D  0.0  0.0  0.0 16.1  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9929
```
The Random Forest model is highly accuracy at 99.29% with a 0.71% error rate on the hold-out data used in the training data set.

Next, the SVM model is to be fit. In addition for SVM, normalizing (via center & scale) the data is important to lower the collinearity/correlation when fitting. The “tuneLength” option will adjust the grid to involve more unique model combinations

### SVM model

```r
fit.svm<-train(classe~.,data=training,method="svmRadial", preProcess=c("center","scale"),
         tuneLength=15,trControl=fitControl)  
confusionMatrix(fit.svm)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.0  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.1 17.3  0.2  0.0
##          D  0.0  0.0  0.1 16.2  0.0
##          E  0.0  0.0  0.0  0.1 18.3
##                             
##  Accuracy (average) : 0.9942
```

```r
stopCluster(cluster) #shuts down parallel processing
```
The SVM fitted model is highly accurate at 99.42% with a 0.58% error rate on the hold-out data used in the training data set. Next we will see if the validate data set shows similar results on the these models.

## Validate new models

```r
pred.rf<-predict(fit.rf,validate[,-54])  # Random Forest model predicting
rf.conf<-confusionMatrix(pred.rf,validate$classe)$overall
rf.conf[1:4]
```

```
##      Accuracy         Kappa AccuracyLower AccuracyUpper 
##     0.9908238     0.9883909     0.9877406     0.9932992
```

```r
pred.svm<-predict(fit.svm,validate[,-54]) # SVM model predicting
svm.conf<-confusionMatrix(pred.svm,validate$classe)$overall
svm.conf[1:4]
```

```
##      Accuracy         Kappa AccuracyLower AccuracyUpper 
##     0.9944943     0.9930355     0.9919995     0.9963687
```
The validate data set, split out of the original training set, shows both Random Forest & SVM models are very accurate with 99.08% accuracy (0.92% error rate) and 99.45% accuracy (0.55% error rate) respectively. In addition, SVM has notably higher Kappa & 95% C.I.. So SVM is the best model to use.

### Plot Best Model

```r
g<-ggplot(fit.svm) 
g+ggtitle("Cost vs Accuracy-SVM model")
```

![](Excersize_Activity1_files/figure-html/model plots-1.png)<!-- -->

```r
plot(varImp(fit.svm),top=7, main="Most Important Variables per classe -SVM model")
```

![](Excersize_Activity1_files/figure-html/model plots-2.png)<!-- -->

Plots show "cost" above 500 doesn't increase accuracy and top seven important variables for the fit SVM model.

# 20 test predictions

```r
pred.20<-predict(fit.svm,newdata=test.tidy)
```

# Appendix
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.     Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4VVOPVcvn
