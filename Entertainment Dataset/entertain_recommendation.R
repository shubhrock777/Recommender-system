
library(recommenderlab)
library(reshape2)


# Load the dataset
library(readr)

####### Example: Data generated in class #####
entertainment <-read.csv(file.choose())
head(entertainment)

entertainment<-entertainment[,2:4]

#details of df
summary(entertainment)



#shape
dim(entertainment)


## covert to matrix format

ratings_matrix <- as.matrix(acast(entertainment, Category~Titles, fun.aggregate = mean))

dim(ratings_matrix)


## recommendarlab realRatingMatrix format
R <- as(ratings_matrix, "realRatingMatrix")

#bulding recommendation


rec = Recommender(R, method="POPULAR")


## create n recommendations for a user

uid = "Action, Adventure, Fantasy"

Titles_ <- subset(entertainment, entertainment$Category==uid)

print("You have rated:")
Titles_

print("recommendations for you:")

#########predication 

prediction <- predict(rec, R[uid], n=5) ## you may change the model here
as(prediction, "list")

