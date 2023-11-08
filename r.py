import pandas 
import numpy
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

movies = pandas.read_csv("tmdb_5000_movies.csv")
credits = pandas.read_csv("tmdb_5000_credits.csv")

# print(movies.head())
# print(credits.head(1)["cast"].values)
# print(credits.head(1)["crew"].values)

# print(movies.merge(credits, on="title").shape)

movies_merge = movies.merge(credits, on = "title")
# print(movies_merge.head())
# print(movies_merge.info())

# Columns needed: genre, movie_id, keywords,  original title, overview, cast, crew
movies_cleaned = movies_merge[["movie_id", "genres", "keywords", "title", "overview", "cast", "crew"]]
# print(movies_cleaned.info()).py

# To find the missing data 
# print(movies_cleaned.isnull().sum())
# Drop the rows with missing columns
movies_cleaned.dropna(inplace=True)

# To check for duplicated columns 
# print(movies_cleaned.duplicated().sum()) 

# Transformation 
# Print first row of genres 
# print(movies_cleaned.iloc[0].genres)
# Output:[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# Required Output: ["Action", "Adventure", "Fantasy"....] 

# Transform genres column 
def convert(obj):
    li = []
    for i in ast.literal_eval(obj):
        li.append(i["name"])
    return li

# print(movies_cleaned.genres.apply(convert))
movies_cleaned["genres"] = movies_cleaned.genres.apply(convert)

# print(movies_cleaned)
# Transform keywords column
movies_cleaned["keywords"] = movies_cleaned.keywords.apply(convert)
# print(movies_cleaned.keywords)

# Transform cast column 
def convert_cast(obj):
    li = []
    count = 0
    for i in ast.literal_eval(obj):
        if(count < 3):
            li.append(i["name"])
            count += 1
        else:
            break
       
    return li 

# print(movies_cleaned['cast'].apply(convert_cast))
movies_cleaned["cast"] = movies_cleaned.cast.apply(convert_cast)

# Transform crew column  
def convert_crew(obj):
    li = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            li.append(i["name"])
            break
    return li 

# print(movies_cleaned.crew.apply(convert_crew))
movies_cleaned["crew"] = movies_cleaned.crew.apply(convert_crew)

# Transform the overview column from sentence to a list 
def convert_overview(obj):
    # x = ast.literal_eval(obj)
    # print("x", x)
    li = obj.split()
    return li

# print(movies_cleaned.overview.apply(convert_overview))
# print(movies_cleaned.overview.apply(lambda x: x.split()))
# print(movies_cleaned.overview[0].split())
movies_cleaned.overview = movies_cleaned["overview"].apply(lambda x: x.split())

# Replace spaces 
movies_cleaned.genres = movies_cleaned["genres"].apply(lambda x: [i.replace(" ", "") for i in x]) 
movies_cleaned.keywords = movies_cleaned["keywords"].apply(lambda x: [i.replace(" ", "") for i in x]) 
movies_cleaned.cast = movies_cleaned["cast"].apply(lambda x: [i.replace(" ", "") for i in x]) 
movies_cleaned.crew = movies_cleaned["crew"].apply(lambda x: [i.replace(" ", "") for i in x]) 

movies_cleaned["tags"] = movies_cleaned.overview + movies_cleaned.genres + movies_cleaned.keywords + movies_cleaned.cast + movies_cleaned.crew

new_df = movies_cleaned[["movie_id", "title", "tags"]]

# convert tag list into string 
new_df.tags = new_df.tags.apply(lambda x: " ".join(x)) 

# Convert all tags letters to lower case 
new_df.tags = new_df["tags"].apply(lambda x: x.lower())

# Vectorize the data using bag of words method 
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df["tags"]).toarray().shape

# print(cv.get_feature_names_out())

# Stemming 
ps = PorterStemmer()

def stem_text(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
# print(new_df["tags"].apply(stem_text))
new_df["tags"] = new_df.tags.apply(stem_text)
# print(new_df["tags"][0])

# Second vectorization 
cv2 = CountVectorizer(max_features=5000, stop_words='english')
vectors2 = cv2.fit_transform(new_df["tags"]).toarray()

# print(cv2.get_feature_names_out()) 
# calculate cosine distance of each movie from the other one 
# print(cosine_similarity(vectors2))
# print(cosine_similarity(vectors2).shape)
similarity = cosine_similarity(vectors2)

# Writing a recommend function that gives a 5 movies based on similarity  
# print(new_df[new_df["title"] == "Batman Begins"].index[0])
# print(sorted(list(enumerate(similarity[0])), reverse = True, key = lambda x: x[1])[1:6])
def recommend(movie):
    # Access the movie index 
    movie_index = new_df[new_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)
recommend('Batman Begins')

















