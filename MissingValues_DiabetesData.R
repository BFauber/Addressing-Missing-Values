###############################################################
###############################################################
# Title: Pima Indians Diabetes Missing Values Problem  
# Date: 01Nov2016
# Authors@R: person("Ben Fauber", role = c("aut", "cre"))
# Description: Pima Indians Diabetes data set from UCI repository
# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
###############################################################
###############################################################

# install packages
install.packages(c("C50", "caret", "e1071", "ggplot2", "kernlab", "mice", "plyr", "randomForest", "VIM"))

# load libraries
library(C50)
library(caret)
library(e1071)
library(ggplot2)
library(kernlab)
library(mice)
library(plyr)
library(randomForest)
library(VIM)


####################################################################
# loading the data
####################################################################

pima <- read.table("~/Desktop/PimaIndianDiabetesData/MAIN_pima-indians-diabetes.data", sep = ",", na.strings="0.0", strip.white=TRUE, fill=TRUE)

# assign column headers

names(pima) <- c("timesPreg", "glucConc", "bloodPres", "skinThick", "serumInsul", "bmi", "pedigreeFunc", "age", "classVar")

#####
# Attribute Information:
#
# 1. Number of times pregnant
# 2. Plasma glucose concentration, at 2 hours in an oral glucose tolerance test (mM)
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2 hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1), 0 = negative, 1 = positive
#####


####################################################################
# understanding the data using summary stats
####################################################################

str(pima)

summary(pima)
# bmi has 11 NA's
# strong skew on serumInsul due to high-end outliers

table(pima$timesPreg)
# peaks around 1 time, max= 17 times (n=1)

table(pima$classVar)
# 35% are Class 1

pimaSub1 <- subset(pima, (
	pima$glucConc > 0 & pima$bloodPres > 0 &
	pima$skinThick > 0 & pima$serumInsul > 0 &
	pima$bmi > 0 & pima$pedigreeFunc > 0 &
	pima$age > 0))
# n=392 (51%)

# using CARET to examine co-linearity in pimaSub1 data

# convert data.frame to numeric format for analysis 

for(i in c(1:ncol(pimaSub1))) 
{
    pimaSub1[,i] <- as.numeric(as.character(pimaSub1[,i]))
}

pimaSub1.cor <- round(cor(pimaSub1), 2)

print(pimaSub1.cor)
# strongest r = 0.68, age to timesPreg
# also r = 0.66, bmi to skinThick
# strongest meaningful r = 0.58, glucConc to serumInsul
# glucConc has the strongest r = 0.52 to classVar
# age is #2 in r = 0.35 to classVar

# revisit the strongest correlations with plots


####################################################################
# understanding the raw data using plots
####################################################################

# density distribution plots, segmented by classVar

pima$classVar <- factor(pima$classVar)

ggplot(data=pima, aes(x=timesPreg, fill=pima$classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Number of Times Pregnant")

ggplot(data=pima, aes(x=glucConc, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Plasma Glucose Concentration, 2 h p.o. Gluc. Tol. Test (mM)")

ggplot(data=pima, aes(x=bloodPres, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Diastolic blood pressure (mm Hg)")

ggplot(data=pima, aes(x=skinThick, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Triceps skin fold thickness (mm)")
	
ggplot(data=pima, aes(x=serumInsul, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="2 h serum insulin (mu U/mL)")

ggplot(data=pima, aes(x=bmi, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Body mass index (weight in kg/(height in m)^2)")

ggplot(data=pima, aes(x=pedigreeFunc, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Diabetes pedigree function")
	
ggplot(data=pima, aes(x=age, fill=classVar)) + 
	geom_density(alpha=0.4) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Age (years)")	

# density plot of classVar
	
ggplot(pima) + 
	geom_density(aes(x=classVar), 
	color="blue", 
	fill="blue",
	alpha=0.2) +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Class variable (0 or 1)")

# histogram of timesPreg

ggplot(pima) + 
	geom_bar(aes(x=timesPreg), 
	color="blue", 
	fill="blue") +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Number of Times Pregnant")

# futher examining the cor identified in the above section, using the pimaSub1 data

ggplot(pimaSub1, aes(x=age, y=timesPreg)) + 
	geom_point(color="blue") + 
	geom_smooth() +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Age (years)", y="Number of Times Pregnant")

ggplot(pimaSub1, aes(x=skinThick, y=bmi)) + 
	geom_point(color="blue") + 
	geom_smooth() + 
	stat_smooth(method="lm", color="red") +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Triceps skin fold thickness (mm)", 
	y="Body mass index (weight in kg/(height in m)^2)")
# nearly linear relationship, added stat_smooth line
			
ggplot(pimaSub1, aes(x=glucConc, y=serumInsul)) + 
	geom_point(color="blue") + 
	geom_smooth() +
	stat_smooth(method="lm", color="red") +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Plasma Glucose Concentration, 2 h p.o. Gluc. Tol. Test (mM)",
	y="2 h serum insulin (mu U/mL)")
# nearly linear relationship, added stat_smooth line

ggplot(pimaSub1, aes(x=glucConc, y=classVar)) + 
	geom_point(color="blue", position=position_jitter(w=0.1, h=0.1)) + 
	geom_smooth() +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Plasma Glucose Concentration, 2 h p.o. Gluc. Tol. Test (mM)",
	y="Class variable (0 or 1)")


# visualizing the relationship of other pimaSub1 variables to classVar

ggplot(pimaSub1, aes(x=timesPreg, y=classVar)) + 
	geom_point(color="blue", position=position_jitter(w=0.1, h=0.1)) + 
	geom_smooth() +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Number of Times Pregnant",
	y="Class variable (0 or 1)")

ggplot(pimaSub1, aes(x=bmi, y=classVar)) + 
	geom_point(color="blue", position=position_jitter(w=0.1, h=0.1)) + 
	geom_smooth() +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Body mass index (weight in kg/(height in m)^2)",
	y="Class variable (0 or 1)")

ggplot(pimaSub1, aes(x=age, y=classVar)) + 
	geom_point(color="blue", position=position_jitter(w=0.1, h=0.1)) + 
	geom_smooth() +
	labs(title="Pima Indians Diabetes Data") +
	labs(x="Age (years)",
	y="Class variable (0 or 1)")
	
		
####################################################################
# cleaning the data
####################################################################

summary(pima) 
# all values are finite, some NA values present in bmi

# omit all unique identifiers in model data
# none found in data, pass

p <- pima

# notes with raw data indicated that all but timesPreg and 
# classVar cols zero values == NA

# addressing the NA and zero values in all but timesPreg and classVar cols

p$glucConc[p$glucConc == 0] <- NA
p$bloodPres[p$bloodPres == 0] <- NA
p$skinThick[p$skinThick == 0] <- NA
p$serumInsul[p$serumInsul == 0] <- NA
p$bmi[p$bmi == 0] <- NA
p$pedigreeFunc[p$pedigreeFunc == 0] <- NA
p$age[p$age == 0] <- NA

# list the rows that do not have missing values

pComp <- p[complete.cases(p), ]

nrow(pComp) 
# 392 complete case rows (out of 768, 51%)

# list the rows that have one or more missing values

pInc <- p[!complete.cases(p), ]

nrow(pInc) 
#376 incomplete case rows (out of 768, 49%)

# significant percentage of rows are missing at least one value, thus cannot omit 
# rows that contain NA values in analysis without losing ~50% of data


# using MICE to understand missing data patterns

md.pattern(p)

# 192 rows have all data except serumInsul and skinThick
# 140 rows have all data except serumInsul
# 26 rows have all data except serumInsul, skinThick, and bloodPres
# 7 rows have all data except serumInsul, skinThick, bloodPres, and bmi
# 4 rows have all data except glucConc and serumInsul
# 7 rows have all data exept a mixture of 1-3 columns
# no NA's present for timesPreg, pedigreeFunc, age, and classVar
# THUS, addressing the serumInsul NA values can address most of the missing data,
# followed by skinThic and bloodPress

# create a shadow matrix of the NA values and explore NA cor

pS <- p

for(i in c(1:ncol(pS))) 
{
    pS[,i] <- as.numeric(as.character(pS[,i]))
}

pShadow <- as.data.frame(abs(is.na(pS)))

y <- pShadow[which(apply(pShadow, 2, sum) > 0)]

pShadow.cor <- cor(y)
# r = 0.66 for NA cor of serumInsul and skinThick
# r = 0.34 for NA cor of bmi and bloodPres
# r = 0.31 for NA cor of skinThick and bloodPres
# r = 0.22 for NA cor of serumInsul and bloodPres 
# all other r values < 0.15

yy <- cor(pS, pShadow, use="pairwise.complete.obs")
# highest r values are 0.22 and 0.21 for age & skinThick and age & serumInsul 
# THUS it is most likely that the missing values are missing at random 
# or missing at complete random

# unable to revisit research team and investigate cause/rationale for missing data
# must address using imputed methods, will use MICE and VIM

pImp <- mice(p, seed=123)

# address serumInsul as it is the largest contributor to NA values
# age and glucConc are have biggest cor to serumInsul, other parameters 
# have some but lower cor

pFit <- with(pImp, lm(serumInsul ~ age + glucConc + bmi + skinThick + bloodPres))

pPool <- pool(pFit)

summary(pPool)

pI <- complete(pImp, action=3)


# convert data.frame to numeric format for analysis 

for(i in c(1:ncol(pI))) 
{
    pI[,i] <- as.numeric(as.character(pI[,i]))
}

# convert classVar values to meaningful factors

pI$classVar <- factor(pI$classVar)

levels(pI$classVar) <- c("negative", "positive")

table(pI$classVar)
# 500=negative, 268=positive


####################################################################
# building the training and hold-out (test) data sets with CARET
####################################################################

# define an 70%/30% train/test split of the data set

set.seed(123)

pItemp <- createDataPartition(pI[,ncol(pI)], p = 0.7, list = FALSE)

pItrain <- pI[pItemp,]
pItest <- pI[-pItemp,]

# count of total values in each dataset

pItrainN <- nrow(pItrain)*ncol(pItrain)
pItestN <- nrow(pItest)*ncol(pItest)


####################################################################
# establish 10-fold cross-validation parameter for training and tuning models
####################################################################

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)


####################################################################
# train classification models with CARET
####################################################################

# build a generalized linear model (binomial distribution of states/outcomes)

set.seed(123)

ptm <- proc.time()

pI1fit <- train(classVar ~., data=pItrain, metric="Accuracy", trControl=fitControl, method="glm", tuneLength=10, na.action=na.omit)

pI1fitT <- proc.time() - ptm
print(pI1fitT)

#list the predictor column names and summarize the model

predictors(pI1fit)
print(pI1fit)
print(pI1fit$bestTune)

pI1fitImpt <- varImp(pI1fit)

ggplot(pI1fitImpt) +
	labs(title=pI1fit$method)


#####
# build a c5.0 classification model
#####

set.seed(123)

ptm <- proc.time()

pI2fit <- train(classVar ~., data=pItrain, metric="Accuracy", trControl=fitControl, method="C5.0", na.action=na.omit)

pI2fitT <- proc.time() - ptm
print(pI2fitT)

#list the predictor column names and summarize the model

predictors(pI2fit)
print(pI2fit)
print(pI2fit$bestTune)

pI2fitImpt <- varImp(pI2fit)

ggplot(pI2fitImpt) +
	labs(title=pI2fit$method)


#####
# build a RandomForest (RF) classification model
#####

# optimize the mtry RF parameter
# default mtry in RF is sqrt(ncol(training))
# bracket mtry parameters, start with grid min = sMt/2 and grid max = 2*sMt

sMt <- round(sqrt(ncol(pItrain)), digits=0)

rfGrid3 <- expand.grid(.mtry = c(sMt/2, sMt, sMt*2))

set.seed(123)

ptm <- proc.time()

pI3fit <- train(classVar ~., data=pItrain, method="rf", metric="Accuracy", trControl=fitControl, ntree=500, tuneGrid=rfGrid3, importance=TRUE, 
na.action=na.omit)

pI3fitT <- proc.time() - ptm
print(pI3fitT)

predictors(pI3fit)
summary(pI3fit)
print(pI3fit)
print(pI3fit$bestTune)

pI3fitImpt <- varImp(pI3fit)

ggplot(pI3fitImpt) +
	labs(title=pI3fit$method)


###################################################################
# save the optimized classification models parameters to txt or csv files
####################################################################

sink(file="~/Desktop/PimaIndianDiabetesData/output/pI_GLM_Model_Summary.txt")
summary(pI1fit)
sink()

sink(file="~/Desktop/PimaIndianDiabetesData/output/pI_c50_Model_Summary.txt")
summary(pI2fit)
sink()

write.csv(summary(pI3fit), file="~/Desktop/PimaIndianDiabetesData/output/pI_RF_Model_Summary.csv")


####################################################################
# applying the optimized classification models against the TRAINING data
####################################################################

pI1pred <- data.frame(predict(pI1fit, pItrain, interval = "predict", level =0.95))
pI2pred <- data.frame(predict(pI2fit, pItrain, interval = "predict", level =0.95))
pI3pred <- data.frame(predict(pI3fit, pItrain, interval = "predict", level =0.95))


####################################################################
# General Functions for STATS
####################################################################

# rounds numbers if numeric

round_numeric <- function(lst, decimals=2) {
    lapply(lst, function(x) {
        if (is.numeric(x)) {
            x <- round(x, decimals)
        }
        x
        })
}

# summary of model stats using a Confusion Matrix as input
# designed for explicit use with the confusionMatrix() function and binomial distribution

sumMod1 <- function(cm) {
    sumM <- list(acc=cm$overall["Accuracy"], # accuracy (TN+TP)/(TP+FP+TN+FN)
                 pre=cm$byClass["Precision"], # precision TP/(TP+FP)
                 rec=cm$byClass["Recall"], # recall TP/(TP+FN)
                 sens=cm$byClass["Sensitivity"],  # sensitivity = recall
                 spec=cm$byClass["Specificity"])  # specificity TN/(TN+FP)
    round_numeric(sumM)
}


####################################################################
# combine all classification model stats into a single data frame
####################################################################

pICmat1 <- confusionMatrix(pI1pred[,1], pItrain[,ncol(pItrain)])
pICmat2 <- confusionMatrix(pI2pred[,1], pItrain[,ncol(pItrain)])
pICmat3 <- confusionMatrix(pI3pred[,1], pItrain[,ncol(pItrain)])


# summary of TRAINING results and metrics

ModelName <- c("glm TRAIN", "c5.0 TRAIN", "RF TRAIN")

pIModelComp <- as.data.frame(
    rbind(sumMod1(pICmat1),
          sumMod1(pICmat2),
          sumMod1(pICmat3)))

pIModelComp <- data.frame(cbind(ModelName, pIModelComp))
pIModelComp <- pIModelComp[order(pIModelComp$ModelName),]
rownames(pIModelComp) <- NULL

print(pIModelComp)

####################################################################
# apply the optimized classification models against the TESTING (hold-out) dataset
####################################################################

pI1predTest <- data.frame(predict(pI1fit, pItest, interval = "predict", level =0.95))
pI2predTest <- data.frame(predict(pI2fit, pItest, interval = "predict", level =0.95))
pI3predTest <- data.frame(predict(pI3fit, pItest, interval = "predict", level =0.95))


####################################################################
# combine all CLASS model TESTING (hold-out) stats into a single data frame
####################################################################

pICmat1T <- confusionMatrix(pI1predTest[,1], pItest[,ncol(pItest)])
pICmat2T <- confusionMatrix(pI2predTest[,1], pItest[,ncol(pItest)])
pICmat3T <- confusionMatrix(pI3predTest[,1], pItest[,ncol(pItest)])

pIModelCompT <- as.data.frame(
    rbind(sumMod1(pICmat1T),
          sumMod1(pICmat2T),
          sumMod1(pICmat3T)))

ModelNameT <- c("glm TEST", "c5.0 TEST", "RF TEST")
pIModelCompT <- data.frame(cbind(ModelNameT, pIModelCompT))
pIModelCompT <- pIModelCompT[order(pIModelCompT$ModelNameT),]
names(pIModelCompT)[names(pIModelCompT)=="ModelNameT"] <- "ModelName"
rownames(pIModelCompT) <- NULL

print(pIModelCompT)


####################################################################
# measure the model performance on the TRAINING vs TESTING (hold-out) data
####################################################################

# merge the training and testing datasets into one table

pIAllStats <- data.frame(rbind(pIModelComp, pIModelCompT))
pIAllStats <- pIAllStats[order(pIAllStats$ModelName),]
row.names(pIAllStats) <- NULL 

print(pIAllStats)

# write a CSV output file for any downstream use

pIAllStatsS <- pIAllStats
pIAllStatsS <- as.matrix(pIAllStatsS)

write.csv(pIAllStatsS, file="~/Desktop/PimaIndianDiabetesData/output/pIModelsAllStats.csv")


