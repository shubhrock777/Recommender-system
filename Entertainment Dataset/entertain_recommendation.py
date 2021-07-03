# import os
import pandas as pd

# import Dataset 
entertainment_df = pd.read_csv("D:/BLR10AM/Assi/10.recommendation engine/Datasets_Recommendation Engine/Entertainment.csv", encoding = 'utf8')

entertainment=entertainment_df.iloc[ :,[1,2,3]]

entertainment.shape # shape
entertainment.columns
entertainment.genre # genre columns

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image:

#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":entertainment.columns,
                "data types ":entertainment.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni =entertainment.nunique()
col_uni


#details of dataframe
entertainment.describe()
entertainment.info()

#checking for null or na vales 
entertainment.isna().sum()
entertainment.isnull().sum()


########exploratory data analysis

EDA = {"columns_name ":entertainment.columns,
                  "mean":entertainment.mean(),
                  "median":entertainment.median(),
                  "mode":entertainment.mode(),
                  "standard_deviation":entertainment.std(),
                  "variance":entertainment.var(),
                  "skewness":entertainment.skew(),
                  "kurtosis":entertainment.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(entertainment.iloc[:, :])

entertainment.columns
#boxplot for every columns


boxplot = entertainment.boxplot(column=[  'Titles', 'Category'])


from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

#checking null value 
entertainment["Category"].isnull().sum() 


# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entertainment.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #51*34

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

# creating a mapping of entertainment name to index number 
entertainment_index = pd.Series(entertainment.index, index = entertainment['Titles']).drop_duplicates()

entertainment_id = entertainment_index["Georgia (1995)"]
entertainment_id



def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    entertainment_id = entertainment_index[Name]
    
    # Getting the pair wise similarity score for all the entertainment's with that 
    # entertainment
    cosine_scores = list(enumerate(cosine_sim_matrix[entertainment_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[1: topN+1]
    
    # Getting the movie index 
    entertainment_idx  =  [i[0] for i in cosine_scores_N]
    entertainment_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    entertainment_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    entertainment_similar_show["name"] = entertainment.loc[entertainment_idx, "Titles"]
    entertainment_similar_show["Score"] = entertainment_scores
    entertainment_similar_show.reset_index(inplace = True)  
    # entertainment_similar_show.drop(["index"], axis=1, inplace=True)
    print (entertainment_similar_show)
    # return (entertainment_similar_show)

    
# Enter your entertainment and number of entertainment's to be recommended 
get_recommendations("To Die For (1995)", topN = 5)
entertainment_index["To Die For (1995)"]

