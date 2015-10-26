args <- commandArgs(TRUE)
install.packages("neuralnet",dependencies=T)
library("neuralnet")
dataURL<-as.character("http://www.utdallas.edu/~axn112530/cs6375/creditset.csv")
header<-as.logical(T)
d<-read.csv(dataURL,header = header)
# create 10 samples
for(i in 1:10) {
  cat("Running sample ",i,"\n")
  set.seed(123)
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  # which one is the class attribute
  Class<-as.factor(as.integer(6))
  # now create all the classifiers and output accuracy values:
  
  #Decision TREE
  trainFit <- rpart(Class ~. , data = trainData , method = 'class',parms = list(split = 'information'))
  predictedTestFit <- predict(trainFit , testData, type="class")
  AccuracyOnTrainedTree <- sum(testData$Class == predictedTestFit)/length(predictedTestFit)

  #SVM Radial Bias Classification Model
  svmRadialModel <- svm(as.factor(class) ~ ., data = trainingData, kernel = "radial", cost = 10, gamma = 0.1)
  svmRadialPred <- predict(svmRadialModel, testData)
  svmRadialPredResult <- sum(testData$class == svmRadialPred)/length(svmRadialPred)

  #***************NAIVE BAYES************
  nbModel <- naiveBayes(Class ~ ., data = trainingData , na.action = na.omit) 
  nbPredictedResult <- predict(nbModel, testData)
  accuracy <- sum(testData$class == nbPredictedResult)/length(nbPredictedResult)

  #k-NN 3 neighbour Classification Model
  NN3Pred <- knn(trainingData,testData,cl=trainingData$class,k=3)
  NN3PredResult <- sum(testData$class == NN3Pred)/length(NN3Pred)
  
  #LOGISTIC REGRESSION
  logisticFit <- glm(Class~., data = trainingData, family = binomial())
  confint(logisticFit)
  predictionValue<-predict(logisticFit, newdata=testData, type="response")
  threshold=1
  prediction<-sapply(predictionValue, FUN=function(x) if (x>threshold) 1 else 0)
  accuracy <- sum(testData$default10yr==prediction)/length(testData)
  print(accuracy)

  #Nueral NET
  library("neuralnet")
  creditnet <- neuralnet(Class ~. , hidden = 4, lifesign = "minimal",linear.output = FALSE, threshold = 0.1)

  # example of how to output
  # method="kNN" 
  # accuracy=0.9
  # cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
}
