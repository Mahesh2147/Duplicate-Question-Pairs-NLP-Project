# Duplicate-Question-Pairs-NLP-Project

-**Dataset Link:** https://www.kaggle.com/c/quora-question-pairs
1. bow_with_basic_features.ipynb
```
  Data Loading & Cleaning:
  	Reads train.csv, samples 30,000 records, and checks for null/duplicate values.
  Exploratory Data Analysis (EDA):
  	Distribution of duplicate/non-duplicate questions.
  	Analyzes repeated questions.
  Feature Engineering:
  	Extracts length of questions, word counts, and shared words.
  	Computes word overlap ratio.
  Bag of Words Model:
  	Uses CountVectorizer to convert text to numerical features.
  	Merges BoW features with engineered features.
  Model Training:
  	Trains Random Forest and XGBoost classifiers.
  	Evaluates models with accuracy scores.
```
2. bow-with-preprocessing-and-advanced-features (1).ipynb
```
  Data Preprocessing:
  	Text cleaning (lowercasing, punctuation removal, stopword filtering).
  Feature Engineering (Advanced):
  	Common word count, total word count, and word share.
  	Token Features (word overlap ratios).
  	Length Features (absolute difference, mean length).
  	Fuzzy Features (Levenshtein distance, fuzzy ratio).
  Visualization:
  	Pairplots of token, length, and fuzzy features.
  Dimensionality Reduction:
  	Uses t-SNE for visualization.
  BoW Implementation:
  	Converts cleaned text into vectors using CountVectorizer.
  Model Training & Evaluation:
  	Trains Random Forest and XGBoost classifiers.
  	Computes confusion matrices.
  Model Deployment:
  	Implements a function to query new pairs.
  	Saves models using pickle.
```
3. initial_EDA.ipynb
```
  Basic Data Exploration:
  	Reads train.csv, checks dataset shape and missing values.
  	Analyzes duplicate and non-duplicate question distributions.
  Repeated Questions Analysis:
  	Examines frequency of repeated questions.
  	Plots histogram of repeated question counts.
```
4. only_bag_of_word.ipynb
```
  BoW-based Feature Engineering:
  	Extracts text from question1 and question2.	
  	Uses CountVectorizer to create Bag of Words representations.
  Model Training:
  	Trains Random Forest and XGBoost classifiers.
  	Evaluates models with accuracy scores.
```
Insights and Conclusion from the Notebooks
1. Insights from the Notebooks

```
  🔹 Data Insights
  	The dataset contains duplicate and non-duplicate question pairs, with class imbalance (more non-duplicate pairs).
  	There are repeated questions, indicating that certain questions are frequently asked in different contexts.
  	Missing values are minimal, so no major data imputation was required.
  	Some questions are very short, while others are lengthy, influencing similarity detection.

  🔹 Feature Engineering Insights
    Basic Features:
    	Question length and word count show a difference between duplicate and non-duplicate pairs.
    	Word overlap ratio (common words / total words) is higher in duplicate questions.
  
    Advanced Features (Token, Length, and Fuzzy Features in bow-with-preprocessing-and-advanced-features):
      Common words count (cwc_min, cwc_max): Higher for duplicate questions.
  	  Character and word-based distance measures (Levenshtein, Fuzzy Features) show that duplicate questions have 	more           similarity.
      First and last word matching between two questions is a good predictor of duplication.
  
  🔹 Bag of Words (BoW) Model Insights
    	BoW captures word frequency but lacks an understanding of meaning/context.
    	High-dimensional feature space can cause sparsity issues in training models.
    	Combining BoW with handcrafted features improves prediction accuracy.
  
  🔹 Model Insights
      Random Forest and XGBoost were tested:
  	  XGBoost performed better due to its ability to capture complex relationships.
  	  Random Forest had lower accuracy, likely due to overfitting on sparse BoW data.
  
  Feature selection impacts accuracy:
    	Only BoW-based models perform moderately well but are improved by additional features.
    	Using token, length, and fuzzy features improves model generalization.
```
2. Conclusion:
   
  i.Feature engineering significantly boosts model performance.
  ii.BoW is a good starting point, but it lacks deep semantic understanding (TF-IDF or Word Embeddings like Word2Vec/BERT        could improve results).
  iii.XGBoost is the best-performing model, making it a strong candidate for final deployment.





