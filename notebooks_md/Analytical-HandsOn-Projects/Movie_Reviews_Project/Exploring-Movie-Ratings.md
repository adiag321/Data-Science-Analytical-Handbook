# Exploring the World of Movies: An Adventure with the IMDb Dataset and DuckDB

Welcome to an exciting journey into the world of movies! In this project, we'll dive deep into the Internet Movie Database (IMDb) dataset using the power and speed of DuckDB, a cutting-edge analytical database. Get ready to uncover fascinating insights about movies, genres, ratings, and more! We'll use Python along with popular data manipulation and visualization libraries to guide us through this adventure.

## Tools for Our Adventure

*   **IMDb Dataset:** A treasure trove of information about movies, TV shows, and their creators. We'll use a subset of the dataset, which is made available for non-commercial use. The dataset is broken down into several files:
    *   `title.akas.tsv.gz`: Contains data about various titles including translated names.
    *   `title.basics.tsv.gz`: Provides basic information for each title, such as type, genres, and runtime.
    *   `title.ratings.tsv.gz`: Includes ratings and vote counts for titles.
    *   `name.basics.tsv.gz`: Contains information about people involved in the creation of titles.
*   **DuckDB:** An in-process SQL OLAP database management system, designed to be fast and efficient for analytical queries.
*   **Python:** Our trusty programming language for orchestrating this data analysis.
*   **Pandas:** A powerful library for data manipulation and analysis. We'll use it to handle data in a tabular format (DataFrames).
*   **Matplotlib:** A foundational plotting library for creating static visualizations.
*   **Seaborn:** A statistical data visualization library built on top of Matplotlib, offering a high-level interface for drawing attractive and informative graphics.

## Our Journey's Path

Our adventure will be divided into several exciting stages:

1. **Getting Started:** We'll set up our tools and get ready for the journey.
2. **Loading the Data:** We'll load the IMDb data directly into DuckDB.
3. **Data Exploration:** We'll delve into each dataset individually, asking questions and uncovering insights with SQL queries and visualizations.
4. **Data Modeling:** We'll create a unified view of the data by joining the individual datasets.
5. **Cross-Dataset Analytical Analysis:** We'll perform more complex analyses, combining data from multiple datasets to reveal deeper patterns and relationships.
6. **Further Exploration Ideas:** We'll brainstorm some additional analyses that could be performed to continue the adventure.

Let the exploration begin!


```python
# Import necessary libraries
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurations ---
DATA_DIR = "./data"  # Directory where your IMDb data files are located. (Make it what you need)
DATABASE_FILE = "imdb.db"  # Name of the DuckDB database file

# --- Establish a connection to DuckDB ---
# conn = duckdb.connect(DATABASE_FILE) # This should be uncommented in a collab notebook
conn = duckdb.connect() # This is to make the code runnable locally


# --- Function to execute a query and return results as a DataFrame ---
def execute_query(query, connection=conn):
    """
    Executes an SQL query and returns the result as a Pandas DataFrame.

    Parameters:
    query (str): The SQL query to execute.
    connection (duckdb.DuckDBPyConnection): The DuckDB connection object. Defaults to the global 'conn'.

    Returns:
    pandas.DataFrame: The result of the query as a DataFrame.
    """
    try:
        result = connection.execute(query).fetchdf()
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return None


# --- Function to load data into DuckDB ---
def load_data_into_duckdb(conn, table_name, file_path):
    """
    Loads data from a TSV file into a DuckDB table.

    Parameters:
    conn (duckdb.DuckDBPyConnection): The DuckDB connection object.
    table_name (str): The name of the table to create.
    file_path (str): The path to the TSV file.
    """
    print(f"Loading data into DuckDB: {table_name}...")
    query = f"""
        CREATE TABLE '{table_name}' AS
        SELECT *
        FROM read_csv_auto('{file_path}', header=true, sep='\t')
    """
    conn.execute(query)
    print(f"{table_name} loaded")


# --- Load data for each dataset ---
load_data_into_duckdb(conn, "title_akas", f"{DATA_DIR}/title.akas.tsv") # These should be uncommented in a collab notebook
load_data_into_duckdb(conn, "title_basics", f"{DATA_DIR}/title.basics.tsv")
load_data_into_duckdb(conn, "title_ratings", f"{DATA_DIR}/title.ratings.tsv")
load_data_into_duckdb(conn, "name_basics", f"{DATA_DIR}/name.basics.tsv")

# --- Create name_basics_exploded table ---
print("Creating name_basics_exploded table...")
conn.execute(
    """
    CREATE TABLE name_basics_exploded AS
    SELECT
        nconst,
        primaryName,
        birthYear,
        deathYear,
        primaryProfession,
        UNNEST(string_to_array(knownForTitles, ',')) AS knownForTitles_exploded
    FROM name_basics
    """
)
print("name_basics_exploded created")
```


```python
# --- 3.1.1 Unique Titles Analysis ---
print("3.1.1 Unique Titles Analysis")
query_unique_titles = "SELECT COUNT(DISTINCT titleId) as unique_titles FROM title_akas"
result_unique_titles = execute_query(query_unique_titles)

print("Number of unique titles:", result_unique_titles["unique_titles"].iloc[0])

# --- 3.1.2 Region Analysis ---
print("3.1.2 Region Analysis")
query_top_regions = """
    SELECT region, COUNT(*) as title_count
    FROM title_akas
    WHERE region IS NOT NULL
    GROUP BY region
    ORDER BY title_count DESC
    LIMIT 10
"""
result_top_regions = execute_query(query_top_regions)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_top_regions, x="title_count", y="region", palette="viridis")
plt.title("Top 10 Regions by Number of Title Variations")
plt.xlabel("Number of Titles")
plt.ylabel("Region")
plt.show()

# --- 3.1.3 Language Analysis ---
print("3.1.3 Language Analysis")
query_top_languages = """
    SELECT language, COUNT(*) as title_count
    FROM title_akas
    WHERE language IS NOT NULL
    GROUP BY language
    ORDER BY title_count DESC
    LIMIT 10
"""
result_top_languages = execute_query(query_top_languages)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_top_languages, x="title_count", y="language", palette="viridis")
plt.title("Top 10 Languages by Number of Title Variations")
plt.xlabel("Number of Titles")
plt.ylabel("Language")
plt.show()

# --- 3.1.4 Title Types Analysis ---
print("3.1.4 Title Types Analysis")

query_title_types = """
    SELECT types, COUNT(*) as type_count
    FROM title_akas
    WHERE types IS NOT NULL
    GROUP BY types
    ORDER BY type_count DESC
    LIMIT 10
"""

result_title_types = execute_query(query_title_types)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_title_types, x="type_count", y="types", palette="viridis")
plt.title("Top 10 Title Types by Frequency")
plt.xlabel("Number of Titles")
plt.ylabel("Title Type")
plt.show()
```


```python
# --- 3.2.1 Title Type Analysis ---
print("3.2.1 Title Type Analysis")
query_title_types = """
    SELECT titleType, COUNT(*) as title_count
    FROM title_basics
    GROUP BY titleType
    ORDER BY title_count DESC
"""
result_title_types = execute_query(query_title_types)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_title_types, x="title_count", y="titleType", palette="viridis")
plt.title("Distribution of Title Types")
plt.xlabel("Number of Titles")
plt.ylabel("Title Type")
plt.show()

# --- 3.2.2 Adult vs. Non-Adult Titles ---
print("3.2.2 Adult vs. Non-Adult Titles")
query_adult_titles = """
    SELECT isAdult, COUNT(*) as title_count
    FROM title_basics
    GROUP BY isAdult
"""
result_adult_titles = execute_query(query_adult_titles)

plt.figure(figsize=(8, 5))
sns.barplot(data=result_adult_titles, x="isAdult", y="title_count", palette="viridis")
plt.title("Distribution of Adult vs. Non-Adult Titles")
plt.xlabel("isAdult (0 = Non-Adult, 1 = Adult)")
plt.ylabel("Number of Titles")
plt.show()

# --- 3.2.3 Start Year Analysis ---
print("3.2.3 Start Year Analysis")
query_start_years = """
    SELECT startYear, COUNT(*) as title_count
    FROM title_basics
    WHERE startYear IS NOT NULL
    GROUP BY startYear
    ORDER BY startYear
"""
result_start_years = execute_query(query_start_years)

plt.figure(figsize=(12, 6))
sns.histplot(data=result_start_years, x="startYear", bins=30, kde=False, color="skyblue")
plt.title("Distribution of Title Start Years")
plt.xlabel("Start Year")
plt.ylabel("Number of Titles")
plt.show()

# --- 3.2.4 Runtime Analysis ---
print("3.2.4 Runtime Analysis")
query_runtimes = """
    SELECT runtimeMinutes
    FROM title_basics
    WHERE runtimeMinutes IS NOT NULL
"""
result_runtimes = execute_query(query_runtimes)

plt.figure(figsize=(12, 6))
sns.histplot(data=result_runtimes, x="runtimeMinutes", bins=30, kde=True, color="skyblue")
plt.title("Distribution of Title Runtimes")
plt.xlabel("Runtime (minutes)")
plt.ylabel("Number of Titles")
plt.show()

# --- 3.2.5 Genre Analysis ---
print("3.2.5 Genre Analysis")
query_genres = """
    SELECT genres, COUNT(*) AS genre_count
    FROM title_basics
    WHERE genres IS NOT NULL
    GROUP BY genres
    ORDER BY genre_count DESC
    LIMIT 10
"""

result_genres = execute_query(query_genres)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_genres, x="genre_count", y="genres", palette="viridis")
plt.title("Distribution of Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.show()
```


```python
# --- 3.3.1 Ratings Distribution ---
print("3.3.1 Ratings Distribution")
query_ratings = "SELECT averageRating FROM title_ratings"
result_ratings = execute_query(query_ratings)

plt.figure(figsize=(10, 6))
sns.histplot(data=result_ratings, x="averageRating", bins=20, kde=True, color="skyblue")
plt.title("Distribution of Average Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Number of Titles")
plt.show()

# --- 3.3.2 Number of Votes Distribution ---
print("3.3.2 Number of Votes Distribution")
query_votes = "SELECT numVotes FROM title_ratings"
result_votes = execute_query(query_votes)

plt.figure(figsize=(10, 6))
sns.histplot(data=result_votes, x="numVotes", bins=30, kde=True, color="skyblue")
plt.title("Distribution of Number of Votes")
plt.xlabel("Number of Votes")
plt.ylabel("Number of Titles")
plt.show()
```


```python
# --- 3.4.1 Primary Profession Analysis ---
print("3.4.1 Primary Profession Analysis")
query_professions = """
    SELECT primaryProfession, COUNT(*) AS profession_count
    FROM name_basics
    WHERE primaryProfession IS NOT NULL
    GROUP BY primaryProfession
    ORDER BY profession_count DESC
    LIMIT 10
"""
result_professions = execute_query(query_professions)

plt.figure(figsize=(10, 6))
sns.barplot(data=result_professions, x="profession_count", y="primaryProfession", palette="viridis")
plt.title("Distribution of Primary Professions")
plt.xlabel("Number of Individuals")
plt.ylabel("Primary Profession")
plt.show()
```


```python
# --- Create the merged_data_sql table ---
print("Creating merged_data_sql table...")
conn.execute(
    """
    CREATE TABLE merged_data_sql AS
    SELECT
        tb.tconst,
        tb.titleType,
        tb.primaryTitle,
        tb.originalTitle,
        tb.isAdult,
        tb.startYear,
        tb.endYear,
        tb.runtimeMinutes,
        tb.genres,
        tr.averageRating,
        tr.numVotes,
        ta.title AS akas_title,
        ta.region,
        ta.language,
        ta.types AS akas_types,
        nb.primaryName,
        nb.birthYear AS primary_birth_year,
        nb.deathYear AS primary_death_year,
        nb.primaryProfession
    FROM
        title_basics tb
        LEFT JOIN title_ratings tr ON tb.tconst = tr.tconst
        LEFT JOIN title_akas ta ON tb.tconst = ta.titleId
        LEFT JOIN name_basics_exploded nb ON nb.knownForTitles_exploded = ta.titleId
    """
)
print("merged_data_sql table created")
```


```python
# --- Genre Popularity Analysis (SQL) ---
print("5.1.1 Genre Popularity Analysis (SQL)")

query_genre_popularity = """
    SELECT
        genres,
        COUNT(*) AS title_count
    FROM
        merged_data_sql
    WHERE
        genres IS NOT NULL
    GROUP BY
        genres
    ORDER BY
        title_count DESC
    LIMIT 10
"""

result_genre_popularity = execute_query(query_genre_popularity)

plt.figure(figsize=(12, 6))
sns.barplot(x='title_count', y='genres', data=result_genre_popularity, palette='viridis')
plt.title('Number of Titles by Genre (Top 10)')
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()

# --- Average Rating by Genre Calculation (SQL) ---
print("5.1.2 Average Rating by Genre Calculation (SQL)")

query_avg_rating_by_genre = """
    SELECT
        genres,
        AVG(averageRating) AS avg_rating
    FROM
        merged_data_sql
    WHERE
        genres IS NOT NULL
    GROUP BY
        genres
    ORDER BY
        avg_rating DESC
    LIMIT 10
"""

result_avg_rating_by_genre = execute_query(query_avg_rating_by_genre)

# --- Average Rating by Genre Visualization (SQL) ---
print("5.1.3 Average Rating by Genre Visualization (SQL)")

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_rating', y='genres', data=result_avg_rating_by_genre, palette='viridis')
plt.title('Average Rating by Genre (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('Genre')
plt.show()
```


```python
# --- Region Popularity Analysis (SQL) ---
print("5.2.1 Region Popularity Analysis (SQL)")

query_region_popularity = """
    SELECT
        region,
        COUNT(*) AS title_count
    FROM
        merged_data_sql
    WHERE
        region IS NOT NULL
    GROUP BY
        region
    ORDER BY
        title_count DESC
    LIMIT 10
"""

result_region_popularity = execute_query(query_region_popularity)

plt.figure(figsize=(12, 6))
sns.barplot(x='title_count', y='region', data=result_region_popularity, palette='viridis')
plt.title('Number of Titles by Region (Top 10)')
plt.xlabel('Number of Titles')
plt.ylabel('Region')
plt.show()

# --- Average Rating by Region Calculation (SQL) ---
print("5.2.2 Average Rating by Region Calculation (SQL)")

query_avg_rating_by_region = """
    SELECT
        region,
        AVG(averageRating) AS avg_rating
    FROM
        merged_data_sql
    WHERE
        region IS NOT NULL
    GROUP BY
        region
    ORDER BY
        avg_rating DESC
    LIMIT 10
"""

result_avg_rating_by_region = execute_query(query_avg_rating_by_region)

# --- Average Rating by Region Visualization (SQL) ---
print("5.2.3 Average Rating by Region Visualization (SQL)")

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_rating', y='region', data=result_avg_rating_by_region, palette='viridis')
plt.title('Average Rating by Region (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('Region')
plt.show()
```


```python
# --- Title Type Popularity Analysis (SQL) ---
print("5.3.1 Title Type Popularity Analysis (SQL)")

query_title_type_popularity = """
    SELECT
        titleType,
        COUNT(*) AS title_count
    FROM
        merged_data_sql
    WHERE
        titleType IS NOT NULL
    GROUP BY
        titleType
    ORDER BY
        title_count DESC
    LIMIT 10
"""

result_title_type_popularity = execute_query(query_title_type_popularity)

plt.figure(figsize=(12, 6))
sns.barplot(x='title_count', y='titleType', data=result_title_type_popularity, palette='viridis')
plt.title('Number of Titles by Title Type (Top 10)')
plt.xlabel('Number of Titles')
plt.ylabel('Title Type')
plt.show()

# --- Average Rating by Title Type Calculation (SQL) ---
print("5.3.2 Average Rating by Title Type Calculation (SQL)")

query_avg_rating_by_title_type = """
    SELECT
        titleType,
        AVG(averageRating) AS avg_rating
    FROM
        merged_data_sql
    WHERE
        titleType IS NOT NULL
    GROUP BY
        titleType
    ORDER BY
        avg_rating DESC
    LIMIT 10
"""

result_avg_rating_by_title_type = execute_query(query_avg_rating_by_title_type)

# --- Average Rating by Title Type Visualization (SQL) ---
print("5.3.3 Average Rating by Title Type Visualization (SQL)")

plt.figure(figsize=(12, 6))
sns.barplot(x='avg_rating', y='titleType', data=result_avg_rating_by_title_type, palette='viridis')
plt.title('Average Rating by Title Type (Top 10)')
plt.xlabel('Average Rating')
plt.ylabel('Title Type')
plt.show()
```


```python
# --- Scatter Plot: Number of Votes vs. Average Rating (SQL) ---
print("5.4.1 Scatter Plot Number of Votes vs Average Rating (SQL)")

query_votes_vs_rating = """
    SELECT
        numVotes,
        averageRating
    FROM
        merged_data_sql
    WHERE
        numVotes IS NOT NULL AND averageRating IS NOT NULL
"""

result_votes_vs_rating = execute_query(query_votes_vs_rating)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='numVotes', y='averageRating', data=result_votes_vs_rating, alpha=0.5)
plt.title('Number of Votes vs. Average Rating')
plt.xlabel('Number of Votes (log scale)')
plt.ylabel('Average Rating')
plt.xscale('log')  # Use log scale for better visualization of wide range of votes
plt.show()
```


```python
# --- Filtering for High Ratings (SQL) ---
print("5.5.1 Filtering for High Ratings (SQL)")

query_high_ratings = """
    SELECT
        *
    FROM
        merged_data_sql
    WHERE
        averageRating >= 8.0
        AND primaryProfession IS NOT NULL
"""

result_high_ratings = execute_query(query_high_ratings)

# --- Exploding Professions and Counting (SQL) ---
print("5.5.2 Exploding Professions and Counting (SQL)")

# Use the result_high_ratings DataFrame as a subquery in your DuckDB query
query_explode_professions = f"""
    SELECT
        UNNEST(string_to_array(primaryProfession, ',')) AS profession,
        COUNT(*) AS profession_count
    FROM
        ({query_high_ratings}) AS high_ratings
    GROUP BY
        profession
    ORDER BY
        profession_count DESC
    LIMIT 10
"""

result_explode_professions = execute_query(query_explode_professions)

# --- Visualizing Most Common Professions (SQL) ---
print("5.5.3 Visualizing Most Common Professions (SQL)")

plt.figure(figsize=(12, 6))
sns.barplot(x='profession_count', y='profession', data=result_explode_professions, palette='viridis')
plt.title('Most Common Primary Professions Associated with High Ratings (>= 8.0) (Top 10)')
plt.xlabel('Number of Titles')
plt.ylabel('Primary Profession')
plt.show()
```


```python
# --- 6.1 Most Common Title Per Decade ---
print("6.1 Most Common Title Per Decade")

query_titles_per_decade = """
WITH TitleDecade AS (
    SELECT
        tconst,
        primaryTitle,
        (startYear / 10) * 10 AS decade
    FROM
        title_basics
    WHERE startYear IS NOT NULL
),
RankedTitles AS (
    SELECT
        decade,
        primaryTitle,
        COUNT(*) AS title_count,
        ROW_NUMBER() OVER (PARTITION BY decade ORDER BY COUNT(*) DESC) AS rank
    FROM
        TitleDecade
    GROUP BY
        decade, primaryTitle
)
SELECT
    decade,
    primaryTitle,
    title_count
FROM
    RankedTitles
WHERE
    rank = 1
ORDER BY
    decade
"""

result_titles_per_decade = execute_query(query_titles_per_decade)

plt.figure(figsize=(12, 6))
sns.barplot(x='decade', y='title_count', hue='primaryTitle', data=result_titles_per_decade, dodge=False, palette='viridis')
plt.title('Most Common Title Per Decade')
plt.xlabel('Decade')
plt.ylabel('Number of Titles')
plt.legend(title='Title')
plt.show()
```


```python
# --- 6.2 Actors with the Longest Movie Careers ---
print("6.2 Actors with the Longest Movie Careers")

query_longest_careers = """
WITH ActorTitleAppearances AS (
    SELECT
        nb.nconst,
        nb.primaryName,
        tb.startYear,
        tb.titleType
    FROM
        name_basics nb
        JOIN title_basics tb ON nb.nconst = tb.tconst
    WHERE
        nb.primaryProfession LIKE '%actor%' OR nb.primaryProfession LIKE '%actress%'
        AND tb.startYear IS NOT NULL
        AND tb.titleType = 'movie'
),
ActorCareerSpans AS (
    SELECT
        nconst,
        primaryName,
        MIN(startYear) AS careerStart,
        MAX(startYear) AS careerEnd
    FROM
        ActorTitleAppearances
    GROUP BY
        nconst, primaryName
)
SELECT
    primaryName,
    careerStart,
    careerEnd,
    careerEnd - careerStart AS careerLength
FROM
    ActorCareerSpans
ORDER BY
    careerLength DESC
LIMIT 10
"""

result_longest_careers = execute_query(query_longest_careers)

plt.figure(figsize=(12, 6))
sns.barplot(x='careerLength', y='primaryName', data=result_longest_careers, palette='viridis')
plt.title('Actors with the Longest Movie Careers')
plt.xlabel('Career Length (Years)')
plt.ylabel('Actor Name')
plt.show()
```


```python
# --- 6.3 Actors with the Longest Gaps Between Movies ---
print("6.3 Actors with the Longest Gaps Between Movies")

query_longest_gaps = """
WITH ActorMovieYears AS (
    SELECT
        nb.nconst,
        nb.primaryName,
        tb.startYear AS movieYear
    FROM
        name_basics nb
        JOIN title_basics tb ON nb.nconst = tb.tconst
    WHERE
        nb.primaryProfession LIKE '%actor%' OR nb.primaryProfession LIKE '%actress%'
        AND tb.startYear IS NOT NULL
        AND tb.titleType = 'movie'
),
ActorMovieYearGaps AS (
    SELECT
        nconst,
        primaryName,
        movieYear,
        LAG(movieYear, 1, movieYear) OVER (PARTITION BY nconst ORDER BY movieYear) AS prevMovieYear
    FROM
        ActorMovieYears
),
ActorLongestGaps AS (
    SELECT
        nconst,
        primaryName,
        MAX(movieYear - prevMovieYear) AS longestGap
    FROM
        ActorMovieYearGaps
    GROUP BY
        nconst, primaryName
)
SELECT
    primaryName,
    longestGap
FROM
    ActorLongestGaps
ORDER BY
    longestGap DESC
LIMIT 10
"""

result_longest_gaps = execute_query(query_longest_gaps)

plt.figure(figsize=(12, 6))
sns.barplot(x='longestGap', y='primaryName', data=result_longest_gaps, palette='viridis')
plt.title('Actors with the Longest Gaps Between Movies')
plt.xlabel('Longest Gap (Years)')
plt.ylabel('Actor Name')
plt.show()
```

## Further Exploration Ideas

Our adventure doesn't have to end here! There are many more exciting avenues to explore in the IMDb dataset. Here are a few ideas to spark your curiosity:

1. **Explore the `attributes` column in `title.akas`:** Analyze the different attributes associated with titles (e.g., "new," "alternative," "working"). What patterns can you find in the attributes and their relationship to regions, languages, or title types?
2. **Analyze the relationship between runtime and ratings:** Do longer movies tend to have higher ratings? Is there a "sweet spot" for movie runtime?
3. **Investigate the distribution of birth and death years in `name_basics`.** How has the age of actors and other professionals changed over time?
4. **Analyze the relationship between the `ordering` of a title in `title_akas` and other variables.** Do earlier titles in a series have higher ratings or more votes?
5. **Network Analysis of Actors and Directors:** Create a network graph where nodes are actors/directors and edges represent collaborations on movies. You could explore questions like:
    *   Who are the most connected actors or directors?
    *   Are there distinct communities or clusters of actors/directors who often work together?

These are just a few ideas to get you started. With a curious mind and the power of SQL and visualization, you can uncover countless other fascinating insights in the IMDb dataset!
