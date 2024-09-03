# Recommendation System for MovieLens 25M Dataset using SURPRISE library

## Problem Statement:

Use the SURPRISE library to create a recommendation system using data from grouplens. Compare the performance of at least the KNNBasic, SVD, NMF, SlopeOne, and CoClustering algorithms to build the recommendations. Share the results of the investigation and include the results from the cross-validation and a basic description of the dataset.

## Description of the Dataset selected:
I selected the MovieLens 25M Dataset from https://grouplens.org/datasets/movielens/25m/. This dataset contains 25 million ratings, one million tag applications applied to 62,000 movies by 162,000 users. There are a few csv files to choose from. I picked only the `ratings.csv` file that contains User Id, Movie Id, Rating, and Timestamp. Since Timestamp was not useful, I excluded that from the analysis. So I used only User Id, Movie Id, and Rating. 

The total number of records in the dataset is 25 million. The processing time to compute all the SURPRISE recommendation systems would be too costly and so I reduced the total population to two subsets.

Due to compute constraints, I created two smaller datasets to evaluate the SURPRISE models:

10k User Dataset: I created a dataset with 10k Users and 10k Movies which resulted in 1.15 M records. (1148902).

5k User Dataset: I did a second pass with this dataset and created a smaller version of 5k Users and 10k Movies.  This reduced the total record count to 0.56M (563139) records.

## Model Selection: 

I selected the following models for my analysis:

1. SVD : The famous Simon Funk's algorithm to solve Netflix Kaggle Competition

2. NMF : A collaborative filtering algorithm based on Non-negative Matrix Factorization

3. SlopeOne: A simple yet accurate collaborative filtering algorithm.

4. Co-Clustering : A collaborative filtering algorithm based on co-clustering

5. kNNBasic : A basic collaborative filtering algorithm

6. kNNWithMeans - A basic collaborative filtering algorithm, taking into account the mean ratings of each user

7. kNNWithZScore - A basic collaborative filtering algorithm, taking into account the z-score normalization of each user

## Evaluation:

As I said before, I ran two rounds of model evaluation. For each set of data, I ran 100 epochs for SVD, Co-Clustering, and NMF. For all the kNNs, I used k=100. The ask was to run the Cross Validation for each of the 7 models with 5 folds.

Here are the results from the **10k User Dataset**

### SVD:

```
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8591  0.8597  0.8628  0.8597  0.8586  0.8600  0.0015  
MAE (testset)     0.6528  0.6546  0.6562  0.6537  0.6536  0.6542  0.0011  
Fit time          25.96   22.28   24.20   21.36   25.99   23.96   1.89    
Test time         0.85    0.85    0.64    0.87    0.86    0.81    0.09  
=========================================================================
```

### NMF (Non-negative Matrix Factorization): 

```
Evaluating RMSE, MAE of algorithm NMF on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8460  0.8497  0.8508  0.8508  0.8487  0.8492  0.0018  
MAE (testset)     0.6507  0.6534  0.6542  0.6548  0.6535  0.6533  0.0014  
Fit time          13.79   11.00   14.01   17.70   11.54   13.61   2.37    
Test time         0.95    0.97    1.53    0.98    0.94    1.07    0.23    
=========================================================================
```

### SlopeOne: 

```
Evaluating RMSE, MAE of algorithm SlopeOne on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8704  0.8640  0.8676  0.8683  0.8666  0.8674  0.0021  
MAE (testset)     0.6657  0.6615  0.6637  0.6647  0.6629  0.6637  0.0014  
Fit time          7.12    7.17    7.22    7.23    7.25    7.20    0.05    
Test time         15.76   15.95   19.09   15.82   16.11   16.55   1.28    
=========================================================================
```

### Co-Clustering: 

```
Evaluating RMSE, MAE of algorithm CoClustering on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8870  0.8891  0.8977  0.8880  0.8885  0.8901  0.0039  
MAE (testset)     0.6894  0.6912  0.6968  0.6897  0.6908  0.6916  0.0027  
Fit time          26.66   31.45   29.35   26.57   29.68   28.74   1.88    
Test time         0.96    0.65    0.64    0.65    0.94    0.77    0.15    
=========================================================================
```

### kNNBasic: 

```
Evaluating RMSE, MAE of algorithm KNNBasic on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8967  0.8982  0.8973  0.8968  0.8969  0.8972  0.0006  
MAE (testset)     0.6857  0.6850  0.6845  0.6847  0.6842  0.6848  0.0005  
Fit time          15.16   15.17   15.31   15.17   15.27   15.22   0.06    
Test time         74.99   79.68   81.14   79.57   82.06   79.49   2.43    
=========================================================================
```

### kNNWithMeans: 

```
Evaluating RMSE, MAE of algorithm KNNWithMeans on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8788  0.8774  0.8755  0.8771  0.8776  0.8773  0.0011  
MAE (testset)     0.6772  0.6756  0.6746  0.6757  0.6774  0.6761  0.0010  
Fit time          14.60   14.96   14.87   15.15   14.79   14.87   0.18    
Test time         70.43   72.47   73.78   82.09   69.23   73.60   4.53 
=========================================================================
```

### kNNWithZScore: 

```
Evaluating RMSE, MAE of algorithm KNNWithZScore on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8793  0.8740  0.8742  0.8753  0.8767  0.8759  0.0020  
MAE (testset)     0.6740  0.6709  0.6713  0.6721  0.6723  0.6721  0.0011  
Fit time          14.98   15.23   14.95   15.01   15.00   15.04   0.10    
Test time         81.27   81.92   75.73   73.15   80.08   78.43   3.41    
=========================================================================
```

Here are the results from the **5k User Dataset**

### SVD:

```
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8749  0.8711  0.8715  0.8711  0.8716  0.8720  0.0014  
MAE (testset)     0.6679  0.6646  0.6647  0.6658  0.6651  0.6656  0.0012  
Fit time          12.90   11.47   12.97   12.71   12.85   12.58   0.56    
Test time         0.63    0.21    0.63    0.66    0.64    0.55    0.17    
=========================================================================
```

### NMF (Non-negative Matrix Factorization): 

```
Evaluating RMSE, MAE of algorithm NMF on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8605  0.8606  0.8566  0.8585  0.8582  0.8589  0.0015  
MAE (testset)     0.6615  0.6617  0.6593  0.6600  0.6609  0.6607  0.0009  
Fit time          6.48    8.35    8.58    6.49    5.89    7.15    1.09    
Test time         0.77    0.76    0.78    0.17    0.78    0.65    0.24      
=========================================================================
```
### SlopeOne: 

```
Evaluating RMSE, MAE of algorithm SlopeOne on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8693  0.8707  0.8688  0.8657  0.8684  0.8686  0.0016  
MAE (testset)     0.6642  0.6660  0.6645  0.6639  0.6650  0.6647  0.0007  
Fit time          3.36    3.47    3.51    3.51    3.49    3.47    0.06    
Test time         8.16    7.95    7.25    8.31    8.45    8.03    0.42       
=========================================================================
```

### Co-Clustering: 

```
Evaluating RMSE, MAE of algorithm CoClustering on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8907  0.8922  0.8964  0.8909  0.8897  0.8920  0.0023  
MAE (testset)     0.6914  0.6929  0.6958  0.6921  0.6901  0.6925  0.0019  
Fit time          16.95   16.11   13.34   12.50   12.96   14.37   1.80    
Test time         0.66    0.65    0.17    0.64    0.67    0.56    0.19   
=========================================================================
```

### kNNBasic: 


```
Evaluating RMSE, MAE of algorithm KNNBasic on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.9035  0.9037  0.9043  0.9026  0.9019  0.9032  0.0008  
MAE (testset)     0.6920  0.6915  0.6922  0.6913  0.6911  0.6916  0.0004  
Fit time          3.21    3.32    3.21    3.21    3.20    3.23    0.05    
Test time         20.76   19.88   17.32   20.34   18.74   19.41   1.24    
=========================================================================
```

### kNNWithMeans: 

```
Evaluating RMSE, MAE of algorithm KNNWithMeans on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8768  0.8743  0.8767  0.8737  0.8769  0.8757  0.0014  
MAE (testset)     0.6738  0.6738  0.6749  0.6720  0.6746  0.6738  0.0010  
Fit time          3.36    3.36    3.16    3.18    3.18    3.25    0.09    
Test time         21.31   19.27   18.24   20.70   21.09   20.12   1.18
=========================================================================
```

### kNNWithZScore: 

```
Evaluating RMSE, MAE of algorithm KNNWithZScore on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.8774  0.8705  0.8742  0.8729  0.8755  0.8741  0.0023  
MAE (testset)     0.6708  0.6668  0.6699  0.6693  0.6698  0.6693  0.0013  
Fit time          3.38    3.42    3.27    3.32    3.34    3.35    0.05    
Test time         19.57   21.01   18.95   21.92   18.55   20.00   1.27   
=========================================================================
```

The results from the **10k User** is summarized as follows:

<img width="717" alt="Screenshot 2024-09-02 at 7 09 42 PM" src="https://github.com/user-attachments/assets/58846537-c32a-4b06-a4c3-a9de0081a1df">

The results from the **5k Users** is summarized as follows: 

<img width="717" alt="Screenshot 2024-09-02 at 7 09 55 PM" src="https://github.com/user-attachments/assets/4c8536f0-a527-4d45-9fa9-7a8193d66bbc">


## Conclusion: 

NMF provides the best result when you compare the RMSE and MAEs. While the training time is higher, the model performs best when it comes to testing the data. 

SVD and SlopeOne are closer to Co Clustering. 

kNN Basic, kNN Means, and kNN Z Score does a good job but the RMSE and MAEs are higher. Also, it takes a lot of time to compute the results. So if we are going to use this for movie recommendations, the cost of processing and recommending a movie will be much higher for each user.

As part of the implementation, if kNN models are batched and packaged as recommendation for the user ahead of time, then it would serve as a good recommendation system. However, if the solution has to be realtime, kNN performs poorly due to compute time.
