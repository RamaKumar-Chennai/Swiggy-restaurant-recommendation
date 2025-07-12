# Swiggy-restaurant-recommendation
The objective is to build a recommendation system based on restaurant data provided in a CSV file. The system should recommend restaurants to users based on input features such as city, rating, cost, and cuisine preferences. The application will utilize clustering or similarity measures to generate recommendations and display results in an easy-to-use Streamlit interface.



Business Use Cases:
Personalized Recommendations: Help users discover restaurants based on their preferences.
Improved Customer Experience: Provide tailored suggestions to enhance decision-making.
Market Insights: Understand customer preferences and behaviors for targeted marketing.
Operational Efficiency: Enable businesses to optimize their offerings based on popular preferences.


Approach:
The dataset is provided as a CSV file with the following columns:
['id', 'name', 'city', 'rating', 'rating_count', 'cost', 'cuisine',
 'lic_no', 'link', 'address', 'menu']
Categorical: name, city, cuisine
Numerical: rating, rating_count, cost

Data Understanding and Cleaning
Duplicate Removal:  The duplicate rows are identified and dropped.
Handling Missing Values: Rows with missing values are dropped.
Few 
The cleaned data is saved to a new CSV file (cleaned_data.csv).


Data Preprocessing
Encoding: One-Hot Encoding is applied to categorical features (name, city, cuisine).
The encoder is saved as a Pickle file (encoder.pkl).
All features are to be numerical after encoding.
The preprocessed dataset is created(encoded_data.csv).
The indices of cleaned_data.csv and encoded_data.csv should match.


Recommendation Methodology
Clustering or Similarity Measures:
K-Means Clustering or Cosine Similarity is used to identify similar restaurants based on input features.
The encoded dataset is used for computations.


Result Mapping:
The recommendation results (indices) are mapped back to the non-encoded dataset (cleaned_data.csv).


Streamlit Application
An interactive application is built with the following components:
User Input: User preferences are accepted (e.g., city, cuisine, rating,price,etc).
Recommendation Engine: The input is processed,the encoded data is queried and  recommendations are generated.
Output: Recommended restaurants are displayed using cleaned_data.csv.

Results: 
Data Preprocessing
Cleaned Dataset (cleaned_data.csv):
Categorical and numerical features with missing values and duplicates removed.
Encoded Dataset (encoded_data.csv):
Preprocessed numerical dataset with categorical features One-Hot Encoded.
Encoder File (encoder.pkl):
Serialized One-Hot Encoder for Streamlit use.
Recommendation System
Clustering or Similarity-based recommendation engine.
Mapping results from encoded_data.csv to cleaned_data.csv for interpretation.
Streamlit Application
User-friendly interface for input and output.
Clear display of recommendations from the cleaned dataset.





 
