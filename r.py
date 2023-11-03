import pandas 
import numpy
import ast

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

# Continue from 49th minute 














