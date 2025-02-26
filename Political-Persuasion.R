setwd("~/Desktop/Big Data/R_datasets")

# Load the libraries
library(ggplot2)
library(gplots)
library(reshape)
library(GGally)

library(rpart)
library(rpart.plot)
library(randomForest)
library(adabag)
library(caret)

library(pROC)
library(xgboost)
library(Matrix)
library(gains)
library(forecast)

library(forecast)
library(dplyr)
library(NeuralNetTools)
library(neuralnet)

# no scientific notation
options(scipen=999)

# Read the file
Voter <- read.csv("VoterNew1new.csv")
names(Voter)

# Descriptive info of Y1
hist(Voter$Y1, xlab="Y1")
summary(Voter$Y1)

# Descriptive table of Y2
table(Voter$Y2)

# Create smaller data files for Y1 and Y2
myvars1 <- c("Y1","NH_WHITE","M_MAR","ED_4COL","HISP","NH_MULT",
             "MED_AGE","COMM_LT10","COMM_609P")
data1 <- Voter[myvars1]
heatmap.2(cor(data1), Rowv = FALSE, Colv = FALSE, dendrogram = "none", 
          cellnote = round(cor(data1),2), 
          notecol = "black", key = FALSE, trace = 'none',margins = c(10,10))

myvars2 <- c("Y2","NH_WHITE","HH_ND","GENDER_F","VPP_12","PARTY_R")
data2 <- Voter[myvars2]

# Check for categorical variables
str(data1)
str(data2)



# Partition the data
set.seed(1)  
train.index <- sample(c(1:dim(data1)[1]), dim(data1)[1]*0.6)  
train1 <- data1[train.index, ]
valid1 <- data1[-train.index, ]

set.seed(1)  
train.index <- sample(c(1:dim(data2)[1]), dim(data2)[1]*0.6)  
train2 <- data2[train.index, ]
valid2 <- data2[-train.index, ]

data2$Y2=as.factor(data2$Y2)
data2$GENDER_F=as.factor(data2$GENDER_F)
data2$VPP_12=as.factor(data2$VPP_12)
data2$PARTY_R=as.factor(data2$PARTY_R)
str(data2)

##Analysis of Y1
## Model1. CART Model for Y1
Ctree1 <- rpart(Y1 ~ ., data = train1, method = "anova",
                minbucket = 20, maxdepth = 4, cp = 0.001)
print(Ctree1)
prp(Ctree1)

# variable importance 
t(t(Ctree1$variable.importance))

# validation accuracy
accuracy(predict(Ctree1, valid1), valid1$Y1)

# compare RNMSE to the mean as percent
errorT=(7580.933/(mean(data1$Y1)))*100
errorT

## Model2. RandomForest (RF) model for Y1

# RF with all variables, change parameters specify
# mtry: Number of variables randomly sampled as candidates at each split.
# ntree: Number of trees to grow
# nodesize = min size of terminal nodes (# of cases)
# importance: Should importance of predictors be assessed?


rf1 <- randomForest(Y1 ~ ., data = train1, ntree = 500, 
                    mtry = 4, nodesize = 5, importance = TRUE)  
print(rf1)

# show importance measures
round(importance(rf1), 2)
# RMSE for training
# obtain MSE as of last element in rf1$mse
rf1$mse[length(rf1$mse)]
# take square root to calculate RMSE for the model
sqrt(rf1$mse[length(rf1$mse)])

### RMSE for the validation data
predict1<-predict(rf1,valid1)
sqrt(mean((valid1$Y1-predict1)^2))

mean_y1 <- mean(data1$Y1)
mean_y1
rel_rmse <- 100 * (1646.081 / 70417.63)
rel_rmse


## Model3. NeuralNet (NN) model for Y1

#create formula for the model
formulaT<- Y1 ~  NH_WHITE + M_MAR + ED_4COL + 
  HISP + NH_MULT + MED_AGE + COMM_LT10 + 
  COMM_609P
class(formulaT)
formulaT


nnT1<-neuralnet(formulaT,train1, hidden=2, algorithm="rprop+",
                err.fct="sse", rep=2,stepmax=1000000,
                threshold=0.01, linear.output=TRUE)
#plot NN
plot(nnT1,show.weights=FALSE)
plot(nnT1)

# show importance measures
round(importance(rf1), 2)
setwd("~/Desktop")
# variable importance
olden(nnT1,file = "myplot.png")


#predicted from valid data
predT <- neuralnet::compute(nnT1,valid1[2:9])

# predicted from valid
predTT<-predT$net.result

#### actual and predicted
actualTT<-valid1$Y1

#The root mean square error
sqr_err<-(actualTT-predTT)^2
sum(sqr_err)
mean(sqr_err)
sqrt(mean(sqr_err))

# compare RNMSE to the mean as percent
mean(data1$Y1)

errorT=(sqrt(mean(sqr_err)))/(mean(data1$Y1))*100
errorT


##### Y2 #####
##Plot the tree
Btree1 <- rpart(Y2 ~ ., data = train2, method = "class", minbucket = 3, maxdepth = 5, cp = 0.001)
print(Btree1)
## variance importance measure
Btree1$variable.importance
prp(Btree1, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
##confusion matrix
PredictTrain <- predict(Btree1,train2,type = "class")
confusionMatrix(PredictTrain,as.factor(train2$Y2))
## ROC & AUROC
train1p <- predict(Btree1,train2,type = "prob")[,2]
tree.roc <- roc(train2$Y2, train1p)
print(tree.roc)
plot(tree.roc)
## Random Forest
rf0 <- randomForest(as.factor(Y2) ~ ., data = train2, ntree = 500, mtry = 4, nodesize = 5, importance= TRUE)  
print(rf0)
## variance importance measure
importance(rf0,type=1)
varImpPlot(rf0, type = 1) 
# Confusion Matrix
rf.pred <- predict(rf0, train2)
confusionMatrix(rf.pred, as.factor(train2$Y2))
rf_prediction <- predict(rf0, train2, type = "prob")
## ROC & AUROC
ROC_rf <- roc(train2$Y2, rf_prediction[,2])
ROC_rf_auc <- auc(ROC_rf)
print(ROC_rf_auc)
plot(ROC_rf)

#create a formula representing the model
formula2<- Y2 ~ NH_WHITE + HH_ND + GENDER_F + VPP_12 + PARTY_R
class(formula2)

nn2<-neuralnet(formula2,train2, hidden=2, algorithm="rprop+",
               err.fct="sse", act.fct="logistic", rep=1,stepmax=1000000,
               threshold=0.5, linear.output=FALSE)

# variable importance
olden(nn2)

#plot NN
plot(nn2,show.weights=FALSE)
plot(nn2)

# NN weights
nn2$result.matrix
nn2$weights
#predicted computation from valid data
predb <- neuralnet::compute(nn2,valid2[2:6]) 
predb

# predicted probability column
predbb<-predb$net.result
predbb

# ROC
ROC1 = roc(valid2$Y2 ~ predbb, plot = TRUE, print.auc = TRUE)

#print AUROC number
print(ROC1)

# Use validation data
## confusion matrix with cutoff point with validation data ###
confusionMatrix(as.factor(ifelse(predbb < 0.5, 0, 1)), as.factor(valid2$Y2))


