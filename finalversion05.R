library(varhandle)
library(pROC)
library(glmnet)
library(MASS)
library(tibble)
library(ResourceSelection)
library(tidyverse)
library(caret)
library(MASS)
library(rpart)
library(rpart.plot)
library(dplyr)
library(car)
library(usdm)
library(e1071)



##1 - Data Preparation
##setwd("/Users/RunzeLeng/Downloads")
##getwd()
data=read.csv("bank-additional-full-updated.csv", header=TRUE)
head(data)
str(data)
dim(data)

data=data[,-11] ##remove variable: duration
str(data)
data=data[,-12] ##remove variable: pdays
str(data)
dim(data)

data$y[data[,19]=="no"]=0
data$y[data[,19]=="yes"]=1 ##change y from yes/no to 1/0
data$y=as.factor(data$y)


#Variance Inflation Factor -Multicollinearity Analysis
data.numeric = dplyr::select_if(data, is.numeric) #selects only numeric columns of a data frame
summary(data.numeric)
str(data.numeric)

cor(data.numeric)
vif(data.numeric)
vifstep(data.numeric,th = 20) ##remove all variables which have vif more than 20

data=data[,-14] ##remove variable: emp.var.rate 
str(data)
data=data[,-16] ##remove variable: euribor3m
str(data)
dim(data)

set.seed(789) ##seperate the data
train.index=sample(c(1:dim(data)[1]),dim(data)[1]*0.5) ##50-50
train.df=data[train.index,]
valid.df=data[-train.index,]
dim(train.df)
dim(valid.df)



##2
##full model
options(scipen=999)
logit.reg.full=glm(y~.,data=train.df,family=binomial(link="logit"))
summary(logit.reg.full)
logit.reg.null=glm(y~1,data=train.df,family=binomial(link="logit"))
summary(logit.reg.null)

##backward and forward selected model with least AIC value of 11372
logit1.reg=stepAIC(logit.reg.null,direction="both",scope=list(upper=logit.reg.full,lower=logit.reg.null))
summary(logit1.reg)
coef(logit1.reg)



##3
##test for full model
logit.reg.pred=predict(logit.reg.full,valid.df[,-17],type="response")
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$predicted>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) ##89.88%

##test for backward and forward model
logit.reg.pred=predict(logit1.reg,valid.df[,-17],type="response")
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$predicted>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) #89.92%



##4 
##LASSO and RIDGE method
x=model.matrix(logit.reg.full)[,-1]
x.test=model.matrix(y~.,valid.df[,])[,-1]
head(x)

##LASSO
cv.lasso = cv.glmnet(x, train.df$y, alpha = 1, family = "binomial")
plot(cv.lasso)
cv.lasso$lambda.min ##0.001048451
cv.lasso$lambda.1se ##0.006140799
opt.lambda=cv.lasso$lambda.min
opt.lambda1=cv.lasso$lambda.1se


##RIDGE
cv.ridge = cv.glmnet(x, train.df$y, alpha = 0, family = "binomial")
plot(cv.ridge)
cv.ridge$lambda.min ##0.01098374
opt.lambda.ridge=cv.ridge$lambda.min


###LASSO TEST WITH lambda.min
logit.reg.full.lasso=glmnet(x, train.df$y, family="binomial", alpha=1)
logit.reg.pred=predict(logit.reg.full.lasso,newx=x.test,type="response",s=opt.lambda)
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$X1>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) #89.87%

#check lasso coefficients
lasso.coef=predict(logit.reg.full.lasso,type="coefficients",s=opt.lambda)
exp(lasso.coef)
sum(lasso.coef!=0) 
sum(lasso.coef==0) 

###LASSO TEST WITH lambda.1se
logit.reg.pred=predict(logit.reg.full.lasso,newx=x.test,type="response",s=opt.lambda1)
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$X1>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) #89.77%


###RIDGE TEST WITH lambda.min
logit.reg.full.ridge=glmnet(x, train.df$y, family="binomial", alpha=0)
logit.reg.pred=predict(logit.reg.full.ridge,newx=x.test,type="response",s=opt.lambda.ridge)
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$X1>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) #89.77%


##5
##Classification Method
valid.y=as.factor(valid.df$y)
CV.ct=rpart(y~.,data=train.df,method="class",cp=0.00001,minsplit=5,xval=5)
printcp(CV.ct)
pruned.ct=prune(CV.ct,cp=CV.ct$cptable[which.min(CV.ct$cptable[,"xerror"]),"CP"])

length(pruned.ct$frame$var[pruned.ct$frame$var=="<leaf>"])
prp(pruned.ct,type=1,extra=1,under=TRUE,split.font=1,varlen=-10,
    box.col=ifelse(pruned.ct$frame$var=="<leaf>",'gray','white'))

pruned.ct.pred.valid=as.factor(predict(pruned.ct,valid.df,type="class"))
confusionMatrix(pruned.ct.pred.valid, valid.y) #89.75%


##6
## Linear Discriminant Analysis
preproc.param <- train.df %>% 
  preProcess(method = c("center", "scale"))

# Transform the data using the estimated parameters
train.transformed <- preproc.param %>% predict(train.df)
test.transformed <- preproc.param %>% predict(valid.df)

model <- lda(y~., data = train.transformed)
# Make predictions
predictions <- model %>% predict(test.transformed)
# Model accuracy
mean(predictions$class==test.transformed$y) ## 88.96%
table(predictions$class,test.transformed$y) 


##7
##Model Lifting
##A(logistic regression)
logit.reg.pred=predict(logit.reg.full,valid.df[,-17],type="response")
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$predicted>0.50,1,0)
a=yes_or_no

##B(reduced model)
##test for backward and forward model
logit.reg.pred=predict(logit1.reg,valid.df[,-17],type="response")
head(logit.reg.pred)
dat=data.frame(actual=valid.df$y,
               predicted=logit.reg.pred)
yes_or_no=ifelse(dat$predicted>0.50,1,0)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(yes_or_no)
confusionMatrix(yes_or_no_class,yes_or_no_actual) #89.92%
b = yes_or_no

##C(discriminant analysis)
class=unfactor(predictions$class)
table(class) 
c=class

##Model Boosting
length(a)
length(b)
length(c)
d=NULL
for(i in 1:20594){
  if((a[i]+b[i]+2*c[i])>=2.5){
    d[i]=1
  }
  else if((a[i]+b[i])>=2){
    d[i]=1
  }
  else{
    d[i]=0
  }
}
a[1]
b[1]
c[1]
head(d)
yes_or_no_actual=as.factor(dat$actual)
yes_or_no_class=as.factor(d)
confusionMatrix(yes_or_no_class,yes_or_no_actual) ##89.91%


