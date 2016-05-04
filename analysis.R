# ------------------------------------------------------------------------------
# INITIAL SETUP
# -------------------------------------------------------------------------------
# setwd
setwd("")

# load library
library(tm)
library(qdap)
library(SnowballC)
library(ROCR)
library(rpart)
library(randomForest)
library(e1071)

# load datasets
Train <- read.csv("train.csv", stringsAsFactors = F)
Test <- read.csv("test.csv", stringsAsFactors = F)

# merge train and test set
Test$Popular <- 2
Comb <- rbind(Train, Test)



# ------------------------------------------------------------------------------
# DATA REFORMAT
# ------------------------------------------------------------------------------
# new vars: extract month, wday, mday, hour, weekend from var PubDate
Comb$PubDate <- strptime(Comb$PubDate, "%Y-%m-%d %H:%M:%S")
Comb$Month <- Comb$PubDate$mon
Comb$Wday <- Comb$PubDate$wday
Comb$Mday <- Comb$PubDate$mday
Comb$Hour <- Comb$PubDate$hour
Comb$Weekend <- ifelse((Comb$Wday == 6 | Comb$Wday == 0),1,0)

# Fill NA values in vars NewsDesk, SectionName, SubsectionName with "Other"
Comb$NewsDesk <- ifelse(Comb$NewsDesk == "", "Other", Comb$NewsDesk)
Comb$SectionName <- ifelse(Comb$SectionName == "", "Other", Comb$SectionName)
Comb$SubsectionName <- ifelse(Comb$SubsectionName == "", "Other", Comb$SubsectionName)

# Factorize categorical data in vars NewsDesk, SectionName, SubsectionName
Comb$NewsDesk <- as.factor(Comb$NewsDesk)
Comb$SectionName <- as.factor(Comb$SectionName)
Comb$SubsectionName <- as.factor(Comb$SubsectionName)



# ------------------------------------------------------------------------------
# STANDARD TEXT PROCESSING
# ------------------------------------------------------------------------------
# build headline corpus
Hcorpus <- Corpus(VectorSource(Comb$Headline))
Hcorpus <- tm_map(Hcorpus, tolower)
Hcorpus <- tm_map(Hcorpus, PlainTextDocument)
Hcorpus <- tm_map(Hcorpus, removePunctuation)
Hcorpus <- tm_map(Hcorpus, removeWords, stopwords("english"))
Hcorpus <- tm_map(Hcorpus, stemDocument, lazy = T)
Hfrequency <- DocumentTermMatrix(Hcorpus)
Hsparse <- removeSparseTerms(Hfrequency, 0.999) # 1064 terms
HeadlineSparse <- as.data.frame(as.matrix(Hsparse))
colnames(HeadlineSparse) = paste0("H_", colnames(HeadlineSparse))
HeadlineSparse$UniqueID = Comb$UniqueID


# build abstract corpus
Acorpus <- Corpus(VectorSource(Comb$Abstract))
Acorpus <- tm_map(Acorpus, tolower)
Acorpus <- tm_map(Acorpus, PlainTextDocument)
Acorpus <- tm_map(Acorpus, removeWords, stopwords("english"))
Acorpus <- tm_map(Acorpus, stemDocument)
Afrequency <- DocumentTermMatrix(Acorpus)
Asparse <- removeSparseTerms(Afrequency, 0.999) # 2123 terms
AbstractSparse <- as.data.frame(as.matrix(Asparse))
colnames(AbstractSparse) = paste0("A_", colnames(AbstractSparse))
AbstractSparse$UniqueID = Comb$UniqueID



# ------------------------------------------------------------------------------
# DATA PRE-PROCESSING
# ------------------------------------------------------------------------------
# new var: polarity 
Comb$Text <- paste0(Comb$Headline, Comb$Abstract, sep = ". ")
pol <- polarity(Comb$Text) # library(qdap)
Comb$Polarity = pol[[1]]$polarity


# merge data frames
Dat <- merge(Comb, HeadlineSparse, by = "UniqueID")
Dat <- merge(Dat, AbstractSparse, by = "UniqueID")


# drop vars not to be included in data modeling
Dat$UniqueID <- NULL
Dat$Headline <- NULL
Dat$Snippet <- NULL
Dat$Abstract <- NULL
# Dat$PubDate <- NULL

# separate dat into train and test set
trainDat <- subset(Dat, Popular != 2)
testDat <- subset(Dat, Popular == 2)

# drop filler value of var popular in test set
testDat$Popular <- NULL


# ------------------------------------------------------------------------------
# DATA MODELING 1: LOGISTIC REGRESSION
# ------------------------------------------------------------------------------
# build model
lrModel <- glm(Popular ~ .,
               data = trainDat,
               family = "binomial")

# make prediction
lrPred <- predict(lrModel,
                  newdata = testDat,
                  type = "response")

# lrPred in data frame
lrPredDF <- data.frame(UniqueID = Test$UniqueID,
                       Probability1 = lrPRed)



# ------------------------------------------------------------------------------
# DATA MODELING 2: RANDOM FOREST
# -----------------------------------------------------------------------
# build model
rfModel <- randomForest(Popular ~ .,
                       data = trainDat,
                       nodesize = 100,
                       ntree = 250)

# make prediction
rfPred <- predict(rfModel,
                  newdata = testDat,
                  type = "class")

# rfPred in data frame
rfPredDF <- data.frame(UniqueID = Test$UniqueID,
                       Probability1 = rfPred)

# ------------------------------------------------------------------------------
# DATA MODELING 3: SUPPORT VECTOR MACHINE
# -----------------------------------------------------------------------
# build model
svmModel <- svm(Popular ~ .,
                data = trainDat)

# make prediction
svmPred <- predict(modelSVM,
                   newdata = testDat)

# svmPred in data frame
svmPredDF <- data.frame(UniqueID = Test$UniqueID,
                        Probability1 = svmPred)


# ------------------------------------------------------------------------------
# DATA MODELING 4: NEURAL NETWORKS
# -----------------------------------------------------------------------
# build model
numFolds = trainControl(method = "cv", number = 10)
nnetGrid = expand.grid(.size = c(6,8),
                       .decay = c(0.008, 0.009, 0.01, 0.011))
nnetModel = train(Popular ~ .,
                  data = trainDat,
                  method = nnet,
                  trControl = numFolds,
                  tuneGrid = nnetGrid)

summary(nnet)

# make prediction
nnetPred <- predict(nnetModel,
                    newdata = trainDat,
                    type = "prob")


# nnetPred in dataframe
nnetPredDF <- data.frame(UniqueID = Test$UniqueID,
                         Probabilty1 = nnetPred)


# ------------------------------------------------------------------------------
# ENSEMBLE LEARNING
# -----------------------------------------------------------------------

