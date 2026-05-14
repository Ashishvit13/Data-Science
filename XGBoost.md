# **XGBoost**
Traditional models like decision trees and random forests are easy to interpret but may lack accuracy on complex data. XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting algorithm that combines multiple weak models into a stronger, high-performance model.

  - It uses decision trees as base learners, building them sequentially so each tree corrects errors from the previous one and it is known as boosting.
  - It features parallel processing for faster training on large datasets and allows parameter customization to optimize performance for specific problems.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/eacf305f-e9e3-4858-814e-87be6b05803a" /><br>

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/dc3e7c6b-addf-494b-9d9c-0aed7f9e2841" /><br>

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/00086c9c-22db-4964-b8c5-5a8a3846427e" /><br>

<img width="801" height="400" alt="image" src="https://github.com/user-attachments/assets/de554e64-b901-45aa-87b7-844b0283fc9c" /><br>

## **How XGBoost Works?**
It builds decision trees sequentially with each tree attempting to correct the mistakes made by the previous one. The process can be broken down as follows:

  1. **Start with a base learner:** The first model decision tree is trained on the data. In regression tasks this base model simply predicts the average of the target variable.
  2. **Calculate the errors:** After training the first tree the errors between the predicted and actual values are calculated.
  3. **Train the next tree:** The next tree is trained on the errors of the previous tree. This step attempts to correct the errors made by the first tree.
  4. **Repeat the process:** This process continues with each new tree trying to correct the errors of the previous trees until a stopping criterion is met.
  5. **Combine the predictions:** The final prediction is the sum of the predictions from all the trees.

## **Mathematics Behind XGBoost Algorithm**
It can be viewed as iterative process where we start with an initial prediction often set to zero. After which each tree is added to reduce errors. Mathematically the model can be represented as:

<img width="395" height="145" alt="image" src="https://github.com/user-attachments/assets/24e505a6-e7a0-4efe-8eec-d67673d396a4" />

The objective function in XGBoost consists of two parts i.e a loss function and a regularization term. The loss function measures how well the model fits the data and the regularization term simplify complex trees. The general form of the loss function is:

<img width="576" height="361" alt="image" src="https://github.com/user-attachments/assets/7ee03364-b98d-4af3-a62d-237909bb5cb2" />

<img width="590" height="404" alt="image" src="https://github.com/user-attachments/assets/6c57aef4-438d-4698-8ddb-93cb88d8e330" />

## **How XGBoost Improves Traditional Gradient Boosting**
XGBoost extends traditional gradient boosting by including regularization elements in the objective function, XGBoost improves generalization and prevents overfitting.

**1. Preventing Overfitting**

XGBoost incorporates several techniques to reduce overfitting and improve model generalization:

  - Learning rate (eta): Controls each tree’s contribution i.e a lower value makes the model more conservative.
  - Regularization: Adds penalties to complexity to prevent overly complex trees.
  - Pruning: Trees grow depth-wise and splits that do not improve the objective function are removed, keeping trees simpler and faster.
  - Combination effect: Using learning rate, regularization and pruning together enhances robustness and reduces overfitting.

**2. Tree Structure**

XGBoost builds trees level-wise (breadth-first) rather than the conventional depth-first approach, adding nodes at each depth before moving to the next level.

  - Best splits: Evaluates every possible split for each feature at each level and selects the one that minimizes the objective function like MSE for regression and cross-entropy for classification.
  - Feature prioritization: Level-wise growth reduces overhead, as all features are considered simultaneously, avoiding repeated evaluations.
  - Benefit: Handles complex feature interactions effectively by considering all features at the same depth.

**3. Handling Missing Data**

XGBoost manages missing values robustly during training and prediction using a sparsity-aware approach.

  - Sparsity-Aware Split Finding: Treats missing values as a separate category when evaluating splits.
  - Default direction: During tree building, missing values follow a default branch.
  - Prediction: Instances with missing features follow the learned default branch.
  - Benefit: Ensures robust predictions even with incomplete input data.

**4. Cache-Aware Access**

XGBoost optimizes memory usage to speed up computations by taking advantage of CPU cache.

  - Memory hierarchy: Frequently accessed data is stored in the CPU cache.
  - Spatial locality: Nearby data is accessed together to reduce memory access time.
  - Benefit: Reduces reliance on slower main memory, improving training speed.

**5. Approximate Greedy Algorithm**

To efficiently handle large datasets, XGBoost uses an approximate method to find optimal splits.

  - Weighted quantiles: Quickly estimate the best split without checking every possibility.
  - Efficiency: Reduces computational overhead while maintaining accuracy.
  - Benefit: Ideal for large datasets where full evaluation is costly.

## **Implementation**

Here we implement XGBoost using Python and the Scikit-learn compatible API to train, predict and evaluate a classification model.

**Step 1: Import Required Libraries**

Import required libraries like.

  - Pandas and NumPy for data manipulation.
  - Matplotlib and Seaborn for visualization.
  - XGBoost with Scikit-learn utilities are used to build and evaluate the classification model.

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import xgboost as xgb
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        from xgboost import XGBClassifier
        
        %matplotlib inline
        sns.set_style("whitegrid")

**Step 2: Load and View the Dataset**

Here, we load the dataset using Pandas and display the first 5 rows to understand its structure, features and sample values.

        df = pd.read_csv("/content/Wholesale customers data.csv")
        df.head()

<img width="616" height="198" alt="image" src="https://github.com/user-attachments/assets/745c844c-6175-46d8-9f47-b367218f32ae" /><br>

**Step 3: Explore Statistical Summary of the Data**

In this step, we use describe() to view key statistics of the dataset which helps in understanding data distribution and spotting anomalies.

        print("\nStatistical Summary")
        display(df.describe())

<img width="897" height="305" alt="image" src="https://github.com/user-attachments/assets/2b1ffc4b-29b2-480e-ba43-346523ed0179" /><br>

**Step 4: Prepare Features and Target, Split Data**

Here, we separate the dataset into features (X) and target labels (y), convert the target into binary format and split the data into training and testing sets for model training and evaluation.

        X = df.drop('Channel', axis=1)
        y = df['Channel'].map({1:1, 2:0})
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

**Step 5: Build and Train the XGBoost Model**

Here we initialize the XGBoost classifier with specified hyperparameters, train it on the training data and make predictions on the test set.

  - Defines the learning objective, tree depth, learning rate, number of trees and regularization to control overfitting.
  - Fits the XGBoost model on the training data (X_train, y_train).
  - Uses the trained model to predict target labels on the test set (X_test).

**Step 6: Evaluate Model Accuracy and Performance**

In this step, we measure how well the XGBoost model performs on the test set using accuracy and a detailed classification report.

        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)
        
        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

<img width="441" height="211" alt="image" src="https://github.com/user-attachments/assets/7eb032f7-600d-46cd-a6bf-1c5e171933fb" /><br>


**Step 7: Plot Confusion Matrix Heatmap**

Visualizes the model’s confusion matrix using a heatmap, helping to quickly identify correct and incorrect predictions.

        plt.figure(figsize=(5,4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

<img width="435" height="396" alt="image" src="https://github.com/user-attachments/assets/a9d595cf-35fb-43da-a00b-c44bdfeceda7" /><br>


**Step 8: Plot Feature Importance**

Here we visualize the importance of each feature in the XGBoost model to understand which variables contribute most to predictions.


        plt.figure(figsize=(8,6))
        xgb.plot_importance(model)
        plt.title("Feature Importance")
        plt.show()

<img width="656" height="440" alt="image" src="https://github.com/user-attachments/assets/02394fc4-94d5-4111-89ea-736469ce0f5a" /><br>


**Step 9: Visualize XGBoost Decision Tree**


Plots one of the trained XGBoost decision trees to help understand how the model makes predictions based on feature splits.
        plt.figure(figsize=(20,10))
        xgb.plot_tree(model, num_trees=0)
        plt.show()

<img width="527" height="297" alt="image" src="https://github.com/user-attachments/assets/b6ccddb1-0b5b-4aae-b0cd-83d3850cf7dc" /><br>


**Advantages**

XGBoost includes several features and characteristics that make it useful in many scenarios:

  - Scalable for large datasets with millions of records.
  - Supports parallel processing and GPU acceleration.
  - Offers customizable parameters and regularization for fine-tuning.
  - Includes feature importance analysis for better insights.
  - Available across multiple programming languages and widely used by data scientists.


**Disadvantages**

XGBoost also has certain aspects that require caution or consideration:

  - Computationally intensive; may not be suitable for resource-limited systems.
  - Sensitive to noise and outliers; careful preprocessing required.
  - Can overfit, especially on small datasets or with too many trees.
  - Limited interpretability compared to simpler models, which can be a concern in fields like healthcare or finance.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/07e23d03-9423-48f2-b064-7f1a7316b94f" /><br>

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/0fcffdb2-caef-4109-ac06-d0a098fe7a09" />
