args <- commandArgs(TRUE)
install.packages("ada",dependencies=T,repos='http://cran.rstudio.com/')
install.packages("neuralnet",dependencies=T,repos='http://cran.rstudio.com/')
install.packages("adabag",dependencies = T,repos='http://cran.rstudio.com/')
install.packages("e1071",dependencies = T,repos='http://cran.rstudio.com/')

library("ada")
library("neuralnet")
library("rpart")
library("adabag")
library("e1071")
library("randomForest")
library("class")

dataURL<-as.character(args[1])
header<-as.logical(args[2])
d<-read.csv(dataURL,header = header,na.strings = c(NA,'?'),skipNul = T)
d<-d[complete.cases(d),]

#VARIABLES DECLARATION FOR ACCURACY PREDICTION SUMMATION 
svmAccuracy <- c(0)
nbSumAccuracy <- c(0)
logitSumAccuracy <- c(0)
id3Accuracy <- c(0)
neuralAccuracy <- c(0)
bagSumAccuracy <- c(0)
boostAccuracy <- c(0)
knnAccuracy <- c(0)
forestAccuracy <- c(0)

# create 10 samples
#set.seed(123)
for(i in 1:10) {
  cat("Running sample ",i)
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData <- d[sampleInstances,]
  testData <- d[-sampleInstances,]

  if("default10yr" == colnames(d)[as.integer(args[3])]){
    #Building the Models for the dataset
    trainFit <- rpart(default10yr~. , data = trainingData , method = 'class',parms = list(split = 'information'))
    svmRadialModel <- svm(as.factor(default10yr) ~ ., data = trainingData, kernel = "radial", cost = 10, gamma = 0.1)    
    nbModel <- naiveBayes(as.factor(default10yr) ~ ., data = trainingData , na.action = na.omit) 
    logisticFit <- glm(default10yr~., data = trainingData, family = binomial())    
    trainingData$default10yr<-as.factor(trainingData$default10yr)
    NN3Pred <- knn(trainingData,testData,cl=trainingData$default10yr,k=9)
    bagModel <- bagging(default10yr~ .,trainingData,mfinal = 10,control = (maxdepth = 1))    
    forestTrainFit <- randomForest(default10yr ~. , data = trainingData,na.action=na.omit)
    netModel <- neuralnet(as.numeric(default10yr) ~ LTI + age, data=trainingData, hidden = 4, lifesign = "minimal",linear.output = FALSE,threshold = 0.1)
    temp_test <- subset(testData, select = c("LTI","age"))
    model <- ada(default10yr ~ ., data = trainingData, iter=20, nu=1, type="discrete")
    threshold=0.6
  }
  else if ("admit" == colnames(d)[as.integer(args[3])]){
    #Building the Models for the dataset
    trainFit <- rpart(admit ~. , data = trainingData , method = 'class',parms = list(split = 'information'))
    svmRadialModel <- svm(as.factor(admit) ~ ., data = trainingData, kernel = "linear", cost = 10, gamma = 0.1)
    nbModel <- naiveBayes(as.factor(admit) ~ ., data = trainingData , na.action = na.omit) 
    logisticFit <- glm(admit~., data = trainingData, family = binomial())
    trainingData$admit<-as.factor(trainingData$admit)
    NN3Pred <- knn(trainingData,testData,cl=trainingData$admit,k=7)
    bagModel <- bagging(admit ~ .,trainingData,mfinal = 20,control = (maxdepth = 2))
    forestTrainFit <- randomForest(admit ~. , data = trainingData,na.action=na.omit)
    model <- ada(admit ~ ., data = trainingData, iter=10, nu=1, type="discrete")
    netModel <- neuralnet(as.numeric(admit)~gre+gpa+rank, trainingData, hidden = 3, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)
    temp_test <- subset(testData, select = c("gre","gpa","rank"))
    threshold=0.55
  }
  else if ("V2" == colnames(d)[as.integer(args[3])]){
   #Removing alphabet labes in the factors in the Data
   if('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'==args[1]){
      levels(testData$V2) <- c(levels(testData$V2),0)
      levels(testData$V2) <- c(levels(testData$V2),1)
      levels(trainingData$V2) <- c(levels(trainingData$V2),0)
      levels(trainingData$V2) <- c(levels(trainingData$V2),1)
      testData$V2[testData$V2=='N'] <- 0
      testData$V2[testData$V2=='R'] <- 1
      trainingData$V2[trainingData$V2=='N'] <- 0
      trainingData$V2[trainingData$V2=='R'] <- 1
    }
    else if('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'==args[1]){
      levels(testData$V2) <- c(levels(testData$V2),0)
      levels(testData$V2) <- c(levels(testData$V2),1)
      levels(trainingData$V2) <- c(levels(trainingData$V2),0)
      levels(trainingData$V2) <- c(levels(trainingData$V2),1)
      testData$V2[testData$V2=='B'] <- 0
      testData$V2[testData$V2=='M'] <- 1      
      trainingData$V2[trainingData$V2=='B'] <- 0
      trainingData$V2[trainingData$V2=='M'] <- 1
    }
    trainingData$V2<-factor(trainingData$V2)
    testData$V2<-factor(testData$V2)

    #Building the Models for the dataset
    trainFit <- rpart(trainingData$V2 ~. , data = trainingData , method = 'class',parms = list(split = 'information'))
    svmRadialModel <- svm(as.factor(trainingData$V2) ~ ., data = trainingData, kernel = "polynomial", cost = 10, gamma = 0.1)
    nbModel <- naiveBayes(as.factor(trainingData$V2) ~ ., data = trainingData , na.action = na.omit) 
    logisticFit <- glm(trainingData$V2~., data = trainingData, family = binomial())
    bagModel <- bagging(V2~ .,trainingData,mfinal = 8,control = (maxdepth = 1))
    NN3Pred <- knn(trainingData,testData,cl=trainingData$V2,k=11)
    forestTrainFit <- randomForest(V2 ~. , data = trainingData,na.action=na.omit)    
    model <- ada(V2 ~ ., data = trainingData, iter=20, nu=1, type="discrete")
    netModel <- neuralnet(as.numeric(V2)~V1+V3+V4+V5+V6+V7+V8+V9+V10+V11+V12+V13+V22+V23+V24+V25+V26+V27+V28+V29+V30+V31+V32, trainingData, hidden = 2, lifesign = "minimal", threshold = 0.5)
    temp_test <- subset(testData, select = c("V1","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V22","V23","V24","V25","V26","V27","V28","V29","V30","V31","V32"))
    threshold=0.6
  }
  else if ("V35" == colnames(d)[as.integer(args[3])]){  
    #Removing alphabet labes in the factors in the Data
    levels(testData$V35) <- c(levels(testData$V35),0)
    levels(testData$V35) <- c(levels(testData$V35),1)
    testData$V35[testData$V35=='b'] <- 0
    testData$V35[testData$V35=='g'] <- 1    
    levels(trainingData$V35) <- c(levels(trainingData$V35),0)
    levels(trainingData$V35) <- c(levels(trainingData$V35),1)
    trainingData$V35[trainingData$V35=='b'] <- 0
    trainingData$V35[trainingData$V35=='g'] <- 1
    trainingData$V35<-factor(trainingData$V35)
    testData$V35<-factor(testData$V35)

    #Building the Models for the dataset
    trainFit <- rpart(trainingData$V35 ~. , data = trainingData , method = 'class',parms = list(split = 'information'))
    svmRadialModel <- svm((trainingData$V35) ~ ., data = trainingData,scale=FALSE, kernel = "sigmoid", cost = 10, gamma = 0.1)
    nbModel <- naiveBayes(as.factor(trainingData$V35) ~ ., data = trainingData , na.action = na.omit) 
    logisticFit <- glm(trainingData$V35~., data = trainingData, family = binomial())
    bagModel <- bagging(V35 ~.,trainingData,mfinal = 9,control = (maxdepth = 1))
    forestTrainFit <- randomForest(V35 ~. , data = trainingData,na.action=na.omit)
    NN3Pred <- knn(trainingData,testData,cl=trainingData$V35,k=9)
    model <- ada(V35 ~ ., data = trainingData, iter=15, nu=1, type="discrete")
    netModel <- neuralnet(as.numeric(V35) ~V1+V3+V4+V5+V6+V7+V8+V9+V11+V13+V14+V15+V16+V17+V18+V19+V21+V23+V24+V25+V26+V27+V28+V29+V31+V30, trainingData, hidden = 4, lifesign = "minimal", threshold = 0.1)
    temp_test <- subset(testData, select = c("V1","V3","V4","V5","V6","V7","V8","V9","V11","V13","V14","V15","V16","V17","V18","V19","V21","V23","V24","V25","V26","V27","V28","V29","V31","V30"))
    threshold=0.65
  }
  
  #PREDICT FROM THE DECISION MODELS FOR 9 TYPES OF CLASSIFIER
  
  #Decision Tree    
  predictedTestFit <- predict(trainFit , testData, type="class")
  #SVM Radial Bias Classification Model
  svmRadialPred <- predict(svmRadialModel, testData)
  #Naive Bayes
  nbPredictedResult <- predict(nbModel, testData)
  #Logistic Regression
  predictionValue<-predict(logisticFit, newdata=testData, type="response")
  prediction<-sapply(predictionValue, FUN=function(x) if (x>threshold) 1 else 0)
  #Nueral net
  Prediction <- compute(netModel, temp_test)
  results <- data.frame(actual = testData[,as.integer(args[3])], prediction = Prediction$net.result)
  #Bagging
  predictedBag <- predict(bagModel , testData)
  #Random Forest
  forestPrediction <- predict(forestTrainFit , testData)
  #ADA Boosting
  p<-predict(model,testData)

  #CALCULATE THE ACCURACY FOR THE CURRENT SET OF DATA VALUES 
  NN3PredResult <- sum(testData[,as.integer(args[3])]== NN3Pred)/length(NN3Pred)
  AccuracyOnTrainedTree <- sum(testData[,as.integer(args[3])] == predictedTestFit)/length(predictedTestFit)
  svmRadialPredResult <- sum(testData[,as.integer(args[3])] == svmRadialPred)/length(svmRadialPred)  
  nbAccuracy <- sum(testData[,as.integer(args[3])] == nbPredictedResult)/length(nbPredictedResult)    
  logitAccuracy <- sum(testData[,as.integer(args[3])]==prediction)/nrow(testData) 
  nueralNetAccuracy <- sum(testData[,as.integer(args[3])]==round(Prediction$net.result))/nrow(testData)
  bagAccuracy <- sum(testData[,as.integer(args[3])] == predictedBag$class)/nrow(testData)  
  AccuracyOnForestValue <- sum(forestPrediction == testData[,as.integer(args[3])])/nrow(testData)
  BoostAccuracy <- sum(testData[,as.integer(args[3])]==p)/length(p)

  #INDIVIDUAL ACCURACY FOR THE DATA SAMPLES
  cat("DECISION TREE Accuracy : ", AccuracyOnTrainedTree,"\n")
  cat("SVM           Accuracy : ", svmRadialPredResult,"\n")
  cat("NB PRED       Accuracy : ", nbAccuracy,"\n")
  cat("KNN           Accuracy : ", NN3PredResult,"\n")
  cat("LOGIT         Accuracy : ", logitAccuracy,"\n")
  cat("Nueral Net    Accuracy : ", 1-nueralNetAccuracy,"\n")
  cat("Bagging       Accuracy : ", bagAccuracy,"\n")
  cat("Random forest Accuracy : ", AccuracyOnForestValue,"\n")
  cat("Boost        Accuracy : ", BoostAccuracy,"\n")

  #CALCULATE SUM OF ALL THE TRIALS IN DATA
  svmAccuracy <- c(svmAccuracy,svmRadialPredResult)
  nbSumAccuracy <- c(nbSumAccuracy,nbAccuracy)
  logitSumAccuracy <- c(logitSumAccuracy,logitAccuracy)
  id3Accuracy <- c(id3Accuracy,AccuracyOnTrainedTree)
  neuralAccuracy <- c(neuralAccuracy,1-nueralNetAccuracy)
  bagSumAccuracy <- c(bagSumAccuracy,bagAccuracy)
  boostAccuracy <- c(boostAccuracy,BoostAccuracy)
  knnAccuracy <- c(knnAccuracy,NN3PredResult)
  forestAccuracy <- c(forestAccuracy,AccuracyOnForestValue)
}
#CALCULATE PREDICTION FOR THE DATA
svmAccuracy <-(sum(svmAccuracy)/10)*100
nbSumAccuracy <- (sum(nbSumAccuracy)/10)*100
logitSumAccuracy <- (sum(logitSumAccuracy)/10)*100
id3Accuracy <- (sum(id3Accuracy)/10)*100
neuralAccuracy <- (sum(neuralAccuracy)/10)*100
bagSumAccuracy <- (sum(bagSumAccuracy)/10)*100
boostAccuracy <- (sum(boostAccuracy)/10)*100
knnAccuracy <- (sum(knnAccuracy)/10)*100
forestAccuracy <- (sum(forestAccuracy)/10)*100

#PRINT THE ACCURACY OF THE DECISION TREE
cat("DECISION TREE Accuracy : ", id3Accuracy,"\n")
cat("SVM           Accuracy : ", svmAccuracy,"\n")
cat("NB PRED       Accuracy : ", nbSumAccuracy,"\n")
cat("KNN           Accuracy : ", knnAccuracy,"\n")
cat("LOGIT         Accuracy : ", logitSumAccuracy,"\n")
cat("Nueral Net    Accuracy : ", neuralAccuracy,"\n")
cat("Bagging       Accuracy : ", bagSumAccuracy,"\n")
cat("Random forest Accuracy : ", forestAccuracy,"\n")
cat("Boost        Accuracy : ", boostAccuracy,"\n")
