
library(recommenderlab)
library(reshape2)


# Load the dataset
library(readr)

####### Example: Data generated in class #####
game_df <-read.csv("D:/BLR10AM/Assi/10.recommendation engine/Datasets_Recommendation Engine/game.csv")
head(game_df)

#details of df
summary(game_df)



#shape
dim(game_df)


## covert to matrix format

ratings_matrix <- as.matrix(acast(game_df, userId~game, fun.aggregate = mean))

dim(ratings_matrix)


## recommendarlab realRatingMatrix format
R <- as(ratings_matrix, "realRatingMatrix")

#bulding recommendation

rec1 = Recommender(R, method="POPULAR")

rec2 = Recommender(R, method="SVD")

## create n recommendations for a user

uid = "34"

game_ <- subset(game_df, game_df$userId==uid)

print("You have rated:")
game_

print("recommendations for you:")

#########predication 

prediction <- predict(rec1, R[uid], n=5) ## you may change the model here
as(prediction, "list")


prediction <- predict(rec2, R[uid], n=5) ## you may change the model here
as(prediction, "list")
