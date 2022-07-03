# fmml20211041
### Foundations of Modern Machine Learning
# SYLLABUS
### LECTURE 1: Introduction to Machine Learning 
1. Video 
### LECTURE 2: Machine Learning Components – Data, Model, Evaluation 
1. Video 
2. Lab 1
3. Lab 2 
### LECTURE 3: Revisiting Nearest Neighbor Classification
1. Video 
2. Lab 1
3. Lab 2 
4. Project 
### LECTURE 4: Retrieval, Performance Evaluation and Metrics  
1. Video
### LECTURE 5: Decision Trees
1. Video 
2. Lab
3. Project 
### LECTURE 6: Linear Classifier 
1. Video 
2. Lab
3. Project
### LECTURE 7: SVM 
1. Video 
2. Lab
3. Project- Byte Panache 
### LECTURE 8: Representing Textual Data 
1. Video 
2. Lab
3. Project
### LECTURE 9: Aadhar: Sequences Matching 
1. Video 
### LECTURE 10: Perceptron and gradient descent 
1. Video 
2. Lab
3. Project
### LECTURE 11: Regression
1. Video 
2. Lab
3. Project 
### LECTURE 12: Clustering 
1. Video 
2. Lab
3. Project 
### LECTURE 13: Feature selection and PCA 
1. Video 
2. Lab
3. Project 
### LECTURE 14: Multilayer perceptron
1. Video 
2. Lab
3. Project 
### LECTURE 15: Neural networks 
1. Video 
2. Lab
3. Project
### LECTURE 16: CNN and Deep Learning 
1. Video 
2. Lab
3. Project
### LECTURE 17: CNN and Deep Learning – 2 
1. Video 
2. Lab
3. Project 
### LECTURE 18: Processing Time Series Data
1. Video 
2. Lab
3. Project 

- - - - 
# DESCRIPTION OF FILES 

## LECTURE 2 LAB 1 :
### Covers basics of python and some libraries such as Numpy, Matplotlib and Nltk.

* Strings, Lists, Tuples, Sets, Dictionaries, and indexing

* Functions

* numpy library: np.array, np.arange, np.eye, np normal multiplication, np point wise multiplication, np transpose, convert even elements of matrix to zero, figure out shape of matrix

* Reading files: open(), read_csv(), perform some operations on dataframes and data visualization, upload image from google drive using imread. 

## LECTURE 2 LAB 2 :
### Introduce data and features 

* Extracting features from data: generating some random data using numpy and performing data visualization 

* Features of text: Download documents from Wikipedia, clean it, ount the frequency of each character in a text and plot it in a histogram,  bigrams, visualize bigrams using heatmap, top 10 ngrams for each text

* Features from written numbers: MNIST dataset, visualize few images, sum of pixel values, visualize sum of pixel values for 0 and 1, count the number black pixels that are surrounded on four sides by non-black pixels, or "hole pixels", visualize, plot for comparision, number of boundary pixels in each image, visualize, plot for comparision

* Features from CSV file: california_housing_train.csv, scatter plot visualization of three features(columns in dataset), exercise: did the same with iris dataset.


## LECTURE 3 LAB 1

### SECTION 1 : distance metrics 
* Euclidean distance
* Manhattan distance 
* Hamming distance
* Cosine similarity 
* Minkowski distance 
* Chebyshev distance 
* Jaccard distance 
* How to decide appropriate distace metric?

### SECTION 2: K-Nearest Neighbor Classifier
* What is KNN?
* What is so unique about KNN?
* implementing your own version of KNN
* sklearn implementation of KNN 
* Weighted NN 
* Understanding Decision Boundaries
* Decision Boundary
* Confusion matrix
* Classification report 
* performance metrics 

### SECTION 2.1: KNN on synthetic dataset

* KNN classifier for synthetic 2-D data 

### SECTION 2.2: KNN on real world dataset

* iris flower dataset or Disher's Iris dataset

### SECTION 3: Feature Normalization

* Imputation of missing values 
* Why is feature normalization required?
* Min-Max scaling 
* Z-score clipping 
* Log normalization
* clipping  

## LECTURE 3 LAB 2 
### Retrieval, Performance Evaluation and Metrics 

* Model Selection 
* Splitting of Dataset 
* Experiment with splits 
* Multiple splits (cross validation) 
* KNN using different train-test split

## LECTURE 3 PROJECT 
### Binary classification of adults 
Use dataset present on Kaggle provided by UCI to perform KNN and find income group 

## LECTURE 5 LAB
### SECTION 1: Classification with Decision Trees 
* Loading IRIS dataset 
* Example of DT on Iris dataset with performance evaluation, and tree structure 
* Entropy and Information 

### SECTION 2:  DT Learning 
* Gini impurity
* Finding pattern out of nowhere 
* Experiment on titanic dataset 

### SECTION 3: Random Forests
* Understanding the motive and idea behid random forests 
* Experiment on Titanic Dataset
* Comparing speed of training of SVMs and Random Forests

## LECTURE 5 PROJECT
* Load data
* Positive datapoints for grids that are finger points 
* Negative datapoints for grids that are not finger points
* Creating a method to cluster the points together and predict the number of fingers based on the number of clusters
* Checking if the clustering is working with the datapoint
* Checking the accuracy over all datapoints 
* Examine on why two of the input images are clustered incorrectly
* Classification using decision trees 
* Classification using Random Forest
* Classification using SVM

## LECTURE 6 LAB
### Linear classifiers and perceptron algorithm

## LECTURE 6 PROJECT 
### Linear Classification
* Implement perceptron linear classification involving perceptron update and classification methods 

## LECTURE 7 LAB
### Support Vector Machines
* Intuitive inroduction to SVMs
* The Kernel Trick
* Kernels: An intuitive explanation

## LECTURE 7 PROJECT 
### SVM classifier
* Data description
* Import Libraries
* Import dataset 
* Exploratory data analysis
* Explore missing values in variables 
* Outliers in numerical variables 
* Handle outliers with SVMs 
* Check the distribution of variables 
* Declare feature vector and target variable 
* Split data into seperate training and test set 
* Feature scaling 
* Run SVM with default hyperparameters
* Run SVM with rbf kernel and C=100.0
* Run SVM with rbf kernel and C=1000.0
* Run SVM with Linear Kernel c=1.0
* Run SVM with Linear Kernel c=100.0
* Run SVM with Linear Kernel c=1000.0
* Compare train-set and test-set accuracy
* Check for overfitting and underfitting 
* Compare model accuracy with null accuracy
* Run SVM with polynomial kernel and C=1.0
* Run SVM with polynomial kernel and C=100.0
* Run SVM with sigmoid kernel and C=1.0
* Run SVM with sigmoid kernel and C=100.0
* Confusion matrix
* Classification metrices
* Classification report
* Classification accuracy
* Classification error
* Precision
* Recall
* True positive rate 
* Fasle positive rate 
* Specificity
* Results and conclusion
* References 

## LECTURE 8 LAB
### Counting for Representation
* Embedding methods
* one-hot encoding 
* Feature vectors
* Data cleaning and preprocessing step 
* Bag of words
* TF-IDF
* Understanding the data: a reviews dataset
* KNN model
* Word2Vec
* References 

## LECTURE 8 PROJECT 
### Representing Textual Data
* imports 
* Data understanding 
* Data pre-processing
* Training the classifiers
* Training Random Forest Classifier 
* Training KNN Classifier 
* Working with Test Data 
* References 

## LECTURE 10 LAB
### Introduction to Gradient Descent 

## LECTURE 10 PROJECT 
### Introduction to Gradient Descent
* Notebook Imports and Packages 
* Example 1 - A simple cost function
* Example 2 - Multiple Minima vs Initial guess and advanced functions 
* Example 3 - Divergence and Overflow 
* Example 4 - Data Visualization with 3D charts

## LECTURE 11 LAB
### Linear Regression
* Brute-force solution
* Gradient Descent 
* Creating data 
* Cost function
* Gradients
* Applying linear regression to housing data 
* Exercises 
* References 
* Further explorations 

## LECTURE 11 PROJECT 
### Covid data analysis with regression 
### PART 1 Data Analysis
* Load the data into pandas dataframe 
* Create a new dataframe which counts the cumulative total number of cases, the cumulative total number of deaths, and also cumulative total number of recoveries for each date.
* Plot the total number of cases per day over time and summarize findings
* Create a new column in the dataframe called “closed cases”
* Create a new column in dataframe called "active cases"
* Create one plot showing trend of number of active cases and closed cases
* Growth factor 
### PART 2 Prediction using linear regression 
* Take the earliest 85% of dates as train and the rest as test 
* We can try different regression and regularizations we have seen before
