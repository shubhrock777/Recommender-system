# import os
import pandas as pd

# import Dataset 
game = pd.read_csv("D:/BLR10AM/Assi/10.recommendation engine/Datasets_Recommendation Engine/game.csv")
game.shape # shape
game.columns
game.game # genre columns

#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":game.columns,
                "data types ":game.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =game.nunique()
col_uni


#details of dataframe
game.describe()
game.info()

#checking for null or na vales 
game.isna().sum()
game.isnull().sum()


########exploratory data analysis

EDA = {"columns_name ":game.columns,
                  "mean":game.mean(),
                  "median":game.median(),
                  "mode":game.mode(),
                  "standard_deviation":game.std(),
                  "variance":game.var(),
                  "skewness":game.skew(),
                  "kurtosis":game.kurt()}

EDA


##histogram


import matplotlib.pyplot as plt

#histogram for rating columns
plt.hist(game.rating)


#boxplot for rating columns
boxplot = game.boxplot(column=['rating'])


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

#checking for null values
game.isnull().sum() 


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000*3068

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of game name to index number 

game_index = pd.Series(game.index, index = game['game']).drop_duplicates()


game_id = game_index["Metroid Prime"]
game_id


def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    game_id = game_index[Name]
    
    # Getting the pair wise similarity score for all the game's with that 
    # game
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[1: topN+1]
    
    # Getting the movie index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    # game_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_similar_show)
    # return (game_similar_show)

    
# Enter your game and number of game's to be recommended 
get_recommendations("Super Mario Galaxy", topN = 10)
game_index["Super Mario Galaxy"]


