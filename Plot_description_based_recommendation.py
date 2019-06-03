
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

#importing the data
orig_df = pd.read_csv(r'..\Downloads\movies_metadata.csv')

df=pd.read_csv('..\Downloads\metadata_cleaned.csv')
df['overview'],df['id']= orig_df['overview'],orig_df['id']

#using TF-IDF vectorizer from sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#Define a TF-IDF vector object. Remove all the english stopwords 
tfidf=TfidfVectorizer(stop_words='english')

#replace NAN with an empty string
df['overview']= df['overview'].fillna('')

#construct the required TF-IDF matrix by applying the fit_transform method on the oveview feature 
tfidf_matrix= tfidf.fit_transform(df['overview'])

#output the shape of tfidf matrix
tfidf_matrix.shape
# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

#reverse mapping for index with movie title

indices=pd.Series(df.index,index=df['title']).drop_duplicates()

#function that takes input as the movie title and output the recommended movies
def content_recommender(title,cosine_sim=cosine_sim,df=df,indices=indices):
    #obtain the index of the movoie that matches the title
    idx=indices[title]
    
    #get the pairwise similarity scores of all the movies with that movie
    #and convert it into a list of tuples
    
    sim_scores=list(enumerate(cosine_sim[idx]))
    
    #sort the movies based on the cosine similarity scores 
    sim_scores= sorted(sim_scores, key=lambda x: x[1],reverse=True)
    
    #scores of top 10 most similar movies ignoring first cz its itself
    
    sim_scores= sim_score[1:11]
    
    #get the indices
    movie_indices= [i[0] for i in sim_scores]
    
    #return top 10 most similar movies
    return df['title'].iloc[movie_indices]
cred_df = pd.read_csv(r'..\Downloads\credits.csv')
key_df = pd.read_csv(r'..\Downloads\keywords.csv')
df['id']=df['id'].astype('int')
# Function to convert all non-integer IDs to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan
    
#Clean the ids of df
df['id'] = df['id'].apply(clean_ids)

#Filter all rows that have a null ID
df = df[df['id'].notnull()]

    # Convert IDs into integer
df['id'] = df['id'].astype('int')
key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')


#converting stringified objects into native python objects 
from ast import literal_eval

features=['cast','crew','keywords','genres']

for feature in features:
    df[feature]=df[feature].apply(literal_eval)

# Extract the director's name. If director is not listed, return NaNdef 
get_director(x):    
    for crew_member in x:        
        if crew_member['job'] == 'Director':            
            return crew_member['name']    
        return np.nan
    
#define the new director feature 
df['Director']= df['crew'].apply(get_director)

#Returns the list of top 3 elements or the entire list
def generate_list(x):
    if isinstance(x,list):
        names= [ele['name'] for ele in x]
        #check if there are more than 3 elements
        if len(names)>3:
            names=names[:3]
        return names
    #return empty list in case there is no element  present
    return []

#apply the generate funcion on the variables
df['cast']= df['cast'].apply(generate_list)
df['keywords']=df['keywords'].apply(generate_list)

df['genres']=df['genres'].apply(lambda x: x[:3])

#function to sanitize the names in the data
def sanitize(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

#apply the sanitize function
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)

#creating a metadata soup

def create_soup(x):    
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df['soup']= df.apply(create_soup,axis=1)

#using CountVectorizer
count=CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

#import cosine similarity function
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#since we droppped few movies based on the bad ids and stuff
#reset the index of the df 
df = df.reset_index()
indices2 = pd.Series(df.index,index=df['title'])

#get the recommendations
content_recommender('The Lion King', cosine_sim2, df,indices2)



    

