#sentiemntal analysis by classifiying textt using naive bayes

####################################
# LOADING ALL THE REQUIRED PACKAGES#
####################################
library(tm)
library(caTools)
library(RTextTools)
library(e1071)
library(openNLP)
library(NLP)
library(dplyr)
library(caret)
library(naivebayes)
library(purrr)
library(dplyr)
library(plyr)
library(stringr)
library(syuzhet)
library(lubridate)
library(ggplot2)
library(stringi)
library(scales)
library(reshape2)
library(qdap)
library(tidyr)
options(max.print = .Machine$integer.max)



########################################
#      READING THE REAL TIME REVIEWS   #      
########################################
file<-read.csv(file.choose(),stringsAsFactors = FALSE)
str(file)
colnames(file)<-c("ItemID","Reviews")
View(file)



#######################################
#  CONVERT CHARACTER INTO FACTOR      #
#######################################
file$ItemID <- as.character(file$ItemID)
file$ItemID<-levels(as.factor(file$ItemID))
file$ItemID




############################
#   BUILDING CORPUS        #
############################
corpus<-Corpus(VectorSource(file$Reviews))
corpus.clean<-corpus
library(SnowballC)



#########################
#  TEXT CLEANING        #
#########################
corpus.clean<-tm_map(corpus.clean,(tolower)) 
corpus.clean<-tm_map(corpus.clean,removePunctuation)  
corpus.clean<-tm_map(corpus.clean,removeNumbers)      

corpus.clean<-tm_map(corpus.clean,removeWords,stopwords()) 
corpus.clean<-tm_map(corpus.clean,stripWhitespace)
clean.text<-corpus.clean
dictCorpus<-corpus.clean
corpus.clean<-tm_map(dictCorpus,stemDocument)
#tokenize the corpus
corpus.cleann<-corpus.clean
myCorpusTokenized <- lapply(corpus.cleann, scan_tokenizer)
#stem complete each token vector
myTokensStemCompleted <- lapply(myCorpusTokenized, stemCompletion, dictCorpus)
myDf <- data.frame(text = sapply(myTokensStemCompleted, paste, collapse = " "), stringsAsFactors = FALSE)
View(myDf)


##########################################
#     SCORE FOR EVERY sentiment          #
##########################################
library(glue)
library(sentimentr)

sentiment<-get_nrc_sentiment(as.character(myDf$text))
View(sentiment)
length(sentiment)
length(file$Reviews)
FinalData <-cbind(file$Reviews,sentiment)
View(FinalData)
finalscore<-get_sentiment(as.character(file$Reviews))
FinalData<-cbind(FinalData,finalscore)
View(FinalData)

##################################################
# calculate total of positive and negative scores#
##################################################
#possum<-sum(FinalData$finalscore>0)
#negsum<-sum(FinalData$finalscore<0)
#f<-possum/sum(FinalData$finalscore)
#f1<-negsum/sum(FinalData$finalscore)


#############################################################
#SPLIT OF CLEAN DATAINTO POsITIVE AND NEGATIVE SENTIMENTS   #
#############################################################
#split the sentences into words...
wordlist<-str_split(myDf$text,'\\s+')
length(wordlist)
#in vector form
words<-unlist(wordlist)
length(words)
View(words)



#########################################################
#   EXTERNAL DATASET OF POSITIVE AND NEGATIVE WORDS     #
#########################################################
pos.words<-scan('C:/Users/Sakshi jain/Desktop/positive_words.txt',what='character',comment.char=':')
neg.words<-scan('C:/Users/Sakshi jain/Desktop/negative_words.txt',what='character',comment.char=':')
#matching positive words
pos.matches<-match(words,pos.words)
length(pos.matches)




#matching negative words
neg.matches<-match(words,neg.words)
length(neg.matches)


#wherever there is NA(no positive or negative value) put false
pos.matches<-!is.na(pos.matches)
neg.matches<-!is.na(neg.matches)


################################################
# WORDS WITH POSITIVE(TRUE) AND NEGATIVE(FALSE)#
################################################
dff<-cbind(as.numeric(pos.matches),as.numeric(neg.matches))
View(dff)

View(table1)


convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
table1<-apply(dff,MARGIN = 2,convert_counts)
table1<-cbind(words,table1)
colnames(table1)<-c("words","positive","negative")
View(table1)


#################################################################
# SCORE OF EVERY SENTIMENT AS POSITIVE(1),NEUTRAL(0),NEGATIVE(-1)    #
#################################################################
finalscore<-pos.matches-neg.matches
#classification of each word as positive(+1),negative(-1) and neutral(0)
b<-ifelse(FinalData$finalscore>=1,"positive sentiment",
ifelse(FinalData$finalscore<=-1,"negative sentiment", "neutral sentiment"))
final<-cbind(FinalData,b)
View(final)




###########################################################
#  NUMBER OF POSITIVE AND NEGATIVE WORDS IN DATASET       #
###########################################################
data<-table(finalscore)
View(data)



###################################################
#  DOCUMENT TERM MATRIX OF CLEAN DATA             #
###################################################
#no of times the particular word appeared in a respective review
library(tm)
dtm <- TermDocumentMatrix(corpus.clean)
m <- as.matrix(dtm)
View(m)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 20)


#############################
#        BAR PLOT           #
#############################
tdm<-TermDocumentMatrix(corpus.clean)
tdm<-as.matrix(tdm)
term_frequency<-rowSums(tdm)
term_frequency<-sort(term_frequency,decreasing=TRUE)
barplot(term_frequency[1:100],col=rainbow(50),las=3)




###################################
#          WORD CLOUD             #                  
##################################
library(wordcloud)
set.seed(1234)
wordcloud(words = names(term_frequency), freq = term_frequency, min.freq = 1,
          max.words=1000, random.order=FALSE, rot.per=0.20, 
          colors=brewer.pal(8, "Dark2"),scale=c(5,0.3))
library(wordcloud2)
term_frequency<-data.frame(names(term_frequency),term_frequency)
colnames(term_frequency)<-c('word','freq')
wordcloud2(term_frequency,shape = 'CIRCLE')

##################################
#SHOWS DISTINCT EMOTIONS         #
#################################
result <- get_nrc_sentiment(as.character(myDf$text))
View(result)
result1<-data.frame(t(result))
new_result <- data.frame(rowSums(result1))
names(new_result)[1] <- "count"
new_result <- cbind("sentiment" = rownames(new_result), new_result)
rownames(new_result) <- NULL
g<-qplot(sentiment, data=new_result[1:10,], weight=count, 
geom="bar",fill=sentiment)+ggtitle("Sentiments")
print(g)
g + theme_bw()


#######################################
#       APPLY THE CLASSIFIER          #
#######################################
#divide file dataset into training and testing
library(caTools)
set.seed(123) 
sample = sample.split(file,SplitRatio = 0.75)
file.train<-subset(file,sample ==TRUE) 
file.test<-subset(file, sample==FALSE)
dim(file.train)
dim(file.test)



clean.corpus.dtm <- DocumentTermMatrix(corpus.clean)
#divide the document term matrix data into training and testing
set.seed(123)
removeSparseTerms(clean.corpus.dtm,0.98)
sample = sample.split(clean.corpus.dtm,SplitRatio = 0.75)

clean.corpus.dtm.train <--subset(as.matrix(clean.corpus.dtm),sample ==TRUE) 
clean.corpus.dtm.test<-subset(as.matrix(clean.corpus.dtm), sample==FALSE)
dim(clean.corpus.dtm.train)
dim(clean.corpus.dtm.test)



#divide the corpus into training and testing
#corpus is converted into dataframe to split
set.seed(123) 
sample = sample.split(data.frame(corpus.clean),SplitRatio = 0.75)
train =subset(corpus.clean,sample ==TRUE)
length(train)
test=subset(corpus.clean, sample==FALSE)
length(test)



#####################################
#       DATA PREPRATION             #
#####################################

#freq_terms<-findFreqTerms((clean.corpus.dtm.train,weighting=TfIdf),lowfreq=3)
#freq_terms
#library(hash)
#library(quanteda)
clean.corpus.dtm.freq.train <- DocumentTermMatrix(train)
dim(clean.corpus.dtm.freq.train)
View(as.matrix(clean.corpus.dtm.freq.train))
clean.corpus.dtm.freq.test  <- DocumentTermMatrix(test)
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
clean.corpus.dtm.freq.train <- apply(clean.corpus.dtm.freq.train, MARGIN = 2, convert_counts)
View(clean.corpus.dtm.freq.train)
clean.corpus.dtm.freq.test  <- apply(clean.corpus.dtm.freq.test, MARGIN = 2, convert_counts)
library(e1071)
#possum<-sum(FinalData$finalscore>0)
#c<-sum(dff[,1])
#postotal<-(c/possum)
#d<-sum(dff[,2])
#negsum<-sum(FinalData$finalscore<0)
#negtotal<-(d/negsum)


#####################################
#   TRAIN THE MODEL                 #
#####################################
library(naivebayes)
library(tidyverse)

text.classifier<-naive_bayes(clean.corpus.dtm.freq.train,as.data.frame(file.train$Reviews))
length(clean.corpus.dtm.freq.train)
length(file.train$user)


text.pred <- predict(text.classifer, clean.corpus.dtm.freq.test[1:44])

CrossTable(text.pred, file.test$Reviews,
           prop.chisq = FALSE, 
           prop.t = FALSE,
           dnn = c('predicted', 'actual'))
