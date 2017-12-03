# Part 1: Individual learners #####################################

library(readr)
library(dplyr)
library(tidyr)
library(mxnet)
library(beepr)
library(h2o)
h2o.init(nthreads = 4)

#set.seed(4321)

train <- read.csv("train.csv")

#changing all vars to numerics:
for(i in 1:ncol(train)){
  train[[i]]<-as.numeric(train[[i]])
}

train<-train[order(train[[1]]),]
TARGET=as.numeric(factor(train[[1]]))-1

train<-as.matrix(cbind(TARGET,train[,-1]))

#TARGET
#head(train)
### Sampling Proportionally ##############################################

#making table with count of each value in target var
count<-train %>% data.frame() %>% group_by(TARGET)%>% summarise(count=n())

#splitting into 10 groups
tab<-cbind(c(1,cumsum(count$count)[-10]),cumsum(count$count))

h <- NULL

for(i in 1:10){
  h<-c(h,sample(tab[i,1]:tab[i,2],3000))
}

valid<-data.frame(TARGET=factor(TARGET[-h]),train[-h,-1]/255) 
train <- data.frame(TARGET=factor(TARGET[h]),train[h,-1]/255)
#will print row.name error^^

#test=test[,!apply(test,2,function(x) sum(x==0))==nrow(test)]
#train=train[,!apply(train,2,function(x) sum(x==0))==nrow(train)]

train<-data.matrix(train)
valid<-data.matrix(valid)

train_x<-train[,-1]
train_y<-train[,1]-1
train_x<-t(train_x)

valid_raw<-valid
valid<-valid[,-1]
valid<-t(valid)

#transforming the data
train.array <- train_x
dim(train.array) <- c(28, 28, 1, ncol(train_x))
valid.array <- valid
dim(valid.array) <- c(28, 28, 1, ncol(valid))

############# Network specifications #############################
#?mx.symbol.Variable


data <- mx.symbol.Variable("data")
devices<-mx.cpu()


#1 convolution
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=relu1, pool_type="max",  kernel=c(2,2), stride=c(2,2))
drop1 <- mx.symbol.Dropout(data=pool1,p=0.5)

#2 convolution 
conv2 <- mx.symbol.Convolution(data=drop1, kernel=c(5,5), num_filter=50)
relu2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=relu2, pool_type="max",    kernel=c(2,2), stride=c(2,2))
drop2 <- mx.symbol.Dropout(data=pool2,p=0.5)

#1 full connected
flatten <- mx.symbol.Flatten(data=drop2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=800)
relu3 <- mx.symbol.Activation(data=fc1, act_type="relu")
drop3 <- mx.symbol.Dropout(data=relu3,p=0.5)

#2 full connected
fc2 <- mx.symbol.FullyConnected(data=drop3, num_hidden=10)

#Final softmax output from network:
mnnet <- mx.symbol.SoftmaxOutput(data=fc2)

mx.set.seed(0)





##### Model training #############################
start <- proc.time()

# standard settings, minima omkring 36 epochs, (V=0.992231404958677, T=0.985633333333335)
model1 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=36, array.batch.size=100,
                                      learning.rate=0.05, momentum=0.8, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100)) #
#[25] Train-accuracy=0.984233333333335
#[25] Validation-accuracy=0.990330578512396


model2 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=38, array.batch.size=100,
                                      learning.rate=0.07, momentum=0.5, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100)) 
#[38] Validation-accuracy=0.991404958677685


model3 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=29, array.batch.size=100,
                                      learning.rate=0.05, momentum=0.8, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100)) 

#[29] Train-accuracy=0.986633333333335
#[29] Validation-accuracy=0.990743801652892

beep()

model4 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=25, array.batch.size=100,
                                      learning.rate=0.1, momentum=0.5, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100)) #


#start <- proc.time() #103 epochs giver: V=0.991322314049586, T=0.990533333333335
#model5 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
#                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
# #                                     ctx=devices, num.round=200, array.batch.size=100,
#                                      learning.rate=0.025, momentum=0.5, wd=0.00001,
#                                      eval.metric=mx.metric.accuracy,                       #[200] Train-accuracy=0.994300000000001
#                                      epoch.end.callback=mx.callback.log.train.metric(100)) #[200] Validation-accuracy=0.992148760330578

model6 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=150, array.batch.size=100,
                                      learning.rate=0.075, momentum=0.5, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,                       #[150] Train-accuracy=0.994566666666668
                                      epoch.end.callback=mx.callback.log.train.metric(100)) #[150] Validation-accuracy=0.99289256198347


model7 <- mx.model.FeedForward.create(mnnet, X=train.array, y=train_y,
                                      eval.data= list(data=valid.array, label=valid_raw[,1]-1),
                                      ctx=devices, num.round=36, array.batch.size=100,
                                      learning.rate=0.05, momentum=0.8, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100)) #[50] Validation-accuracy=0.991652892561983
#[25] Train-accuracy=0.985600000000001
#[25] Validation-accuracy=0.992148760330578
#[36] Train-accuracy=0.987100000000001
#[36] Validation-accuracy=0.993719008264462


#beep();Sys.sleep(1);beep();Sys.sleep(1);beep()
#beep();Sys.sleep(1);beep();Sys.sleep(1);beep()

end <- proc.time()
print(end - start)


############################## Predictions ########################################
# Now we have a bag of models, these need to be predicted, and cbined;


# evaluation on the final model:
#number_of_models = seq(1:6)
#for (i in number_of_models){
#paste("sum(diag(table(valid_raw[,1]-1,pred",i,".label)))/nrow(valid_raw)", sep="")
#print(ii)
#}

pred1 <- predict(model1, valid.array)
pred1.label <- max.col(t(pred1)) - 1
a1 <- sum(diag(table(valid_raw[,1]-1,pred1.label)))/nrow(valid_raw)

pred2 <- predict(model2, valid.array)
pred2.label <- max.col(t(pred2)) - 1
a2 <- sum(diag(table(valid_raw[,1]-1,pred2.label)))/nrow(valid_raw)

pred3 <- predict(model3, valid.array)
pred3.label <- max.col(t(pred3)) - 1
a3 <- sum(diag(table(valid_raw[,1]-1,pred3.label)))/nrow(valid_raw)

pred4 <- predict(model4, valid.array)
pred4.label <- max.col(t(pred4)) - 1
a4 <- sum(diag(table(valid_raw[,1]-1,pred4.label)))/nrow(valid_raw)

pred5 <- predict(model5, valid.array)
pred5.label <- max.col(t(pred5)) - 1
a5 <- sum(diag(table(valid_raw[,1]-1,pred5.label)))/nrow(valid_raw)

pred6 <- predict(model6, valid.array)
pred6.label <- max.col(t(pred6)) - 1
a6 <- sum(diag(table(valid_raw[,1]-1,pred6.label)))/nrow(valid_raw)


# evaluation on the final model:
Accuracy <- cbind(a1,a2,a3,a4,a5,a6)
colnames(Accuracy) = c("Model1","Model2","Model3","Model4","Model5","Model6")
Accuracy
barplot(Accuracy)


#Storing the correct values of the validation frame
validlabels <- valid_raw[,1]-1
validlabels

#Storing the correct values of the training frame
trainlabels <- train_y
trainlabels


#cbinding all the predictions together with correct labels

stackedpredictions <- cbind(pred1.label,pred2.label,pred3.label,pred4.label,pred5.label,pred6.label, validlabels)
head(stackedpredictions)

# Initial test: exporting to do cross-validated prediction in flow
stackedpredictions.h2o <- as.h2o(stackedpredictions, destination_frame = "stackedpredictions.hex")
stackedpredictions.h2o <- as.factor(stackedpredictions.h2o)

h2o.summary(stackedpredictions.h2o)


# Part 3: Ensemble learning on top of data #####################################

response <- "validlabels" #labels if components is present
predictors <- setdiff(names(stackedpredictions.h2o), response)


####### Ensemble deeplearning model ######      current classification err = 0.00630, 6 epochs

#t0 <- proc.time()
metalearner <- h2o.deeplearning(
  model_id="dl_model_tuned", 
  
  training_frame=stackedpredictions.h2o, 
  validation_frame=stackedpredictions.h2o,
  nfolds = 10, #8,10
  
  x=predictors, 
  y=response, 
  activation="RectifierWithDropout",
  overwrite_with_best_model=T,    ## IF false = Return the final model after 10 epochs, even if not the best
  hidden=c(800,400, 200),              ## more hidden layers -> more complex interactions
  balance_classes=T,
  epochs=20,                      ## to keep it short enough
  score_validation_samples=10000, ## downsample validation set for faster scoring
  #score_duty_cycle=0.025,        ## don't score more than 2.5% of the wall time
  adaptive_rate=T,                ## manually tuned learning rate
  #rho=0.001,                     ##Adaptive learning rate time decay factor (similarity to prior updates)
  #epsilon=1.0e-10,               ##Adaptive learning rate parameter, similar to learn rate annealing during initial training phase. Typical values are between 1.0e-10 and 1.0e-4
  input_dropout_ratio=0.25,
  hidden_dropout_ratio=c(0.25,0.25, 0.25), #0.5,0.5
  #loss ="CrossEntropy",  
  #rate=0.001, 
  #rate_annealing=2e-6,            
  momentum_start=0.2,             ## manually tuned momentum
  momentum_stable=0.4, 
  momentum_ramp=1e7, 
  #l1=1e-5,                        ## L1/L2 regularization
  #l2=1e-5,
  max_w2=10                       ## helps stability for Rectifier
) 
summary(metalearner)
beep()
#t1 <- proc.time()
#t = t1-t0
#t
#?h2o.deeplearning








##########################################################
########## Prediction from ensemble learners ##############

test <- read.csv("/users/mikeriess/desktop/MNIST/test.csv")
#will give errrs ^^

#converting testframe into numeric
for(i in 1:ncol(test)){
  test[[i]]<-as.numeric(test[[i]])
}

#normalizing the format of the images
test2<-t(as.matrix(test/255))
dim(test2)<-c(28,28,1,ncol(test2))

# Predicting from all the models:

#pre <-predict(model,test2)
#predtarget_test <- apply(pre,2,function(x) which.max(x))-1


pred1_1 <- predict(model1, test2)
pred1_1.label <- apply(pred1_1,2,function(x) which.max(x))-1

pred2_1 <- predict(model2, test2)
pred2_1.label <- apply(pred2_1,2,function(x) which.max(x))-1

pred3_1 <- predict(model3, test2)
pred3_1.label <- apply(pred3_1,2,function(x) which.max(x))-1

pred4_1 <- predict(model4, test2)
pred4_1.label <- apply(pred4_1,2,function(x) which.max(x))-1

pred5_1 <- predict(model5, test2)
pred5_1.label <- apply(pred5_1,2,function(x) which.max(x))-1

pred6_1 <- predict(model6, test2)
pred6_1.label <- apply(pred6_1,2,function(x) which.max(x))-1


##########################################################
#### Cbinding all the predictions ######################

stackedpredictions2 <- cbind(pred1_1.label,pred2_1.label,pred3_1.label,
                             pred4_1.label,pred5_1.label,pred6_1.label)
colnames(stackedpredictions2) = c("pred1.label","pred2.label","pred3.label",
                                  "pred4.label","pred5.label","pred6.label")
head(stackedpredictions2)
str(stackedpredictions2)
head(stackedpredictions)
str(stackedpredictions)

# Initial test: exporting to do cross-validated prediction in flow
stackedpredictions2.h2o <- as.h2o(stackedpredictions2, 
                                  destination_frame = "stackedpredictions2.hex")
#stackedpredictions2.h2o <- as.factor(stackedpredictions2.h2o)

df.pred = h2o.predict(object = metalearner, newdata = stackedpredictions2.h2o)

#summary(df.pred)

PREDS.df <- as.data.frame(df.pred)
PREDS.df$ImageId = 1:nrow(PREDS.df)

PREDS.df <- PREDS.df[,-2:-11]
#str(PREDS.df)

PREDS.df = PREDS.df[,c(2,1)]
colnames(PREDS.df) <- c("ImageId","label")
str(PREDS.df)

write.csv(PREDS.df ,file="submission_stacked_December30.csv", 
          row.names = FALSE) # col.names = TRUE,


