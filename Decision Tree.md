# **Decision Tree Algorithm**

Decision trees are a simple machine learning tool used for classification and regression tasks. They break complex decisions into smaller steps, making them easy to understand and implement. This article explains all about decision tree Algorithm , how decision trees work, their advantages, disadvantages, and applications.

## **Table of contents**
  1. What is Decision Tree?
  2. Types of Decision Tree
  3. Decision Tree Terminologies
  4. How Decision Tree Algorithms Work?
  5. Decision Tree Assumptions
  6. Advantages of Decision Trees
  7. Disadvantages of Decision Trees
  8. How do Decision Trees use Entropy?
  9. Applications of Decision Trees
  10. Conclusion
  11. Frequently Asked Questions


## **What is Decision Tree?**

A decision tree, which has a hierarchical structure made up of root, branches, internal, and leaf nodes, is a non-parametric supervised learning approach used for classification and regression applications.

It is a tool that has applications spanning several different areas. These trees can be used for classification as well as regression problems. The name itself suggests that it uses a flowchart like a tree structure to show the predictions that result from a series of feature-based splits. It starts with a root node and ends with a decision made by leaves.

<img width="452" height="299" alt="image" src="https://github.com/user-attachments/assets/1ec612dc-c28e-4181-a908-dcf166a80d0f" />

## **Types of Decision Tree**
  - **ID3:** This algorithm measures how mixed up the data is at a node using something called entropy. It then chooses the feature that helps to clarify the data the most.C4.5 : This is an improved version of ID3 that can handle missing data and continuous attributes.
  - **CART:** This algorithm uses a different measure called Gini impurity to decide how to split the data. It can be used for both classification (sorting data into categories) and regression (predicting continuous values) tasks.

## **Decision Tree Terminologies**

Before learning more about decision trees let’s get familiar with some of the terminologies:

  - **Root Node:** The initial node at the beginning of a decision tree, where the entire population or dataset starts dividing based on various features or conditions.
  - **Decision Nodes:** Nodes resulting from the splitting of root nodes are known as decision nodes. These nodes represent intermediate decisions or conditions within the tree.
  - **Leaf Nodes:** Nodes where further splitting is not possible, often indicating the final classification or outcome. Leaf nodes are also referred to as terminal nodes.
  - **Sub-Tree:** Similar to a subsection of a graph being called a sub-graph, a sub-section of a these tree is referred to as a sub-tree. It represents a specific portion of the decision tree.
  - **Pruning:** The process of removing or cutting down specific nodes in a tree to prevent overfitting and simplify the model.
  - **Branch / Sub-Tree:** A subsection of the entire is referred to as a branch or sub-tree. It represents a specific path of decisions and outcomes within the tree.
  - **Parent and Child Node:** In a decision tree, a node that is divided into sub-nodes is known as a parent node, and the sub-nodes emerging from it are referred to as child nodes. The parent node represents a decision or condition, while the child nodes represent the potential outcomes or further decisions based on that condition.

<img width="600" height="290" alt="image" src="https://github.com/user-attachments/assets/927346ff-147a-4a62-b6fe-5c0524b295d4" />

### **Example of Decision Tree**
Let’s understand decision trees with the help of an example:

<img width="422" height="344" alt="image" src="https://github.com/user-attachments/assets/6fbd5001-11d9-4ed2-a0ce-62a982a53316" />
<br>
Decision trees are upside down. This means the root is at the top. Then this root is split into various nodes. They are nothing but a bunch of if-else statements in layman terms. It checks if the condition is true and if it is then it goes to the next node attached to that decision.

In the below diagram the tree will first ask what is the weather? Is it sunny, cloudy, or rainy? If yes then it will go to the next feature which is humidity and wind. It will again check if there is a strong wind or weak. If it’s a weak wind and it’s rainy, the person may go and play.

<img width="600" height="265" alt="image" src="https://github.com/user-attachments/assets/8cadc3f6-b7ad-45d6-b6c8-c2c84a830913" />
<br>
  - Did you notice anything in the above flowchart? We see that if the weather is cloudy then we must go to play. Why didn’t it split more? Why did it stop there?
  - To answer this question, we need to know about few more concepts like entropy, information gain, and Gini index. But in simple terms, I can say here that the output for the training dataset is always yes for cloudy weather. Since there is no disorderliness here, we don’t need to split the node further.
  - The goal of machine learning is to decrease uncertainty or disorders from the dataset and for this, we use these trees.
  - Now you must be thinking how do I know what should be the root node? what should be the decision node? when should I stop splitting? To decide this, there is a metric called “Entropy” which is the amount of uncertainty in the dataset.

 ## **How Decision Tree Algorithms Work?**
Decision Tree algorithm works in simpler steps:

  - **Starting at the Root:** The algorithm begins at the top, called the “root node,” representing the entire dataset.
  - **Asking the Best Questions:** It looks for the most important feature or question that splits the data into the most distinct groups. This is like asking a question at a fork in the tree.
  - **Branching Out:** Based on the answer to that question, it divides the data into smaller subsets, creating new branches. Each branch represents a possible route through the tree.
  - **Repeating the Process:** The algorithm continues asking questions and splitting the data at each branch until it reaches the final “leaf nodes,” representing the predicted outcomes or classifications.

## **Decision Tree Assumptions**
Several assumptions are made to build effective models when creating decision trees. These assumptions help guide the tree’s construction and impact its performance. Here are some common assumptions and considerations when creating decision trees:

**1. Binary Splits**
Decision trees typically make binary splits, meaning each node divides the data into two subsets based on a single feature or condition. This assumes that each decision can be represented as a binary choice.

**2. Recursive Partitioning**
Decision trees use a recursive partitioning process, where each node is divided into child nodes, and this process continues until a stopping criterion is met. This assumes that data can be effectively subdivided into smaller, more manageable subsets.

**3. Feature Independence**
These trees often assume that the features used for splitting nodes are independent. In practice, feature independence may not hold, but it can still perform well if features are correlated.

**4. Homogeneity**
It aim to create homogeneous subgroups in each node, meaning that the samples within a node are as similar as possible regarding the target variable. This assumption helps in achieving clear decision boundaries.

**5. Top-Down Greedy Approach**
They are constructed using a top-down, greedy approach, where each split is chosen to maximize information gain or minimize impurity at the current node. This may not always result in the globally optimal tree.

## **Advantages of Decision Trees**
  - **Easy to Understand:** They are simple to visualize and interpret, making them easy to understand even for non-experts.
  - **Handles Both Numerical and Categorical Data:** They can work with both types of data without needing much preprocessing.
  - **No Need for Data Scaling:** These trees do not require normalization or scaling of data.
  - **Automated Feature Selection:** They automatically identify the most important features for decision-making.
  - **Handles Non-Linear Relationships:** They can capture non-linear patterns in the data effectively.

## **Disadvantages of Decision Trees**
  - **Overfitting Risk:** It can easily overfit the training data, especially if they are too deep.
  - **Unstable with Small Changes:** Small changes in data can lead to completely different trees.
  - **Biased with Imbalanced Data:** They tend to be biased if one class dominates the dataset.
  - **Limited to Axis-Parallel Splits:** They struggle with diagonal or complex decision boundaries.
  - **Can Become Complex:** Large trees can become hard to interpret and may lose their simplicity.

## **How do Decision Trees use Entropy?**
Decision Tree used Entropy in the given Points :

  - Now we know what entropy is and what is its formula, Next, we need to know that how exactly does it work in this algorithm.
  - Entropy basically measures the impurity of a node. Impurity is the degree of randomness; it tells how random our data is. Apure sub-splitmeans that either you should be getting “yes”, or you should be getting “no”.
  - Supposea featurehas 8 “yes” and 4 “no” initially, after the first split the left node gets 5 ‘yes’ and 2 ‘no’whereas right node gets 3 ‘yes’ and 2 ‘no’.
  - We see here the split is not pure, why? Because we can still see some negative classes in both the nodes. In order to make a this tree, we need to calculate the impurity of each split, and when the purity is 100%, we make it as a leaf node.

To check the impurity of feature 2 and feature 3 we will take the help for Entropy formula.

<img width="600" height="309" alt="image" src="https://github.com/user-attachments/assets/c83ffb2c-eac1-4ec5-862c-74268adad2b5" />

<img width="574" height="305" alt="image" src="https://github.com/user-attachments/assets/8769fb6c-1978-404f-89e7-7887c9910e68" />

For feature 3,
<img width="578" height="296" alt="image" src="https://github.com/user-attachments/assets/ab0afebc-5c2d-48a4-98e8-ca611df2efbf" />

  - We can clearly see from the tree itself that left node has low entropy or more purity than right node since left node has a greater number of “yes” and it is easy to decide here.
  - Always remember that the higher the Entropy, the lower will be the purity and the higher will be the impurity.
  - As mentioned earlier the goal of machine learning is to decrease the uncertainty or impurity in the dataset, here by using the entropy we are getting the impurity of a particular node, we don’t know if the parent entropy or the entropy of a particular node has decreased or not.
  - For this, we bring a new metric called “Information gain” which tells us how much the parent entropy has decreased after splitting it with some feature.

## **Applications of Decision Trees**
  **1. Healthcare**
  - Diagnosing diseases based on patient symptoms: Decision trees help doctors analyze symptoms and medical history to identify potential illnesses. For example, they can determine if a patient has diabetes or heart disease by evaluating factors like age, weight, and test results.
  - Predicting patient outcomes and treatment effectiveness: Decision trees can predict how a patient might respond to a specific treatment, helping doctors choose the best course of action.
  - Identifying risk factors for specific health conditions: They can analyze data to find patterns, such as lifestyle habits or genetic factors, that increase the risk of diseases like cancer or diabetes.

**2. Finance**
  - Assessing credit risk for loan approvals: Decision trees evaluate an applicant’s credit history, income, and other factors to decide whether to approve or reject a loan application.
  - Detecting fraudulent transactions: By analyzing transaction patterns, decision trees can flag unusual or suspicious activities, helping banks prevent fraud.
  - Predicting stock market trends and investment risks: They analyze historical data to forecast market trends, helping investors make informed decisions.

**3. Marketing**
  - Segmenting customers for targeted campaigns: Decision trees group customers based on their behavior, preferences, or demographics, allowing businesses to create personalized marketing strategies.
  - Predicting customer churn and retention: They analyze customer data to identify those likely to stop using a service, enabling companies to take proactive steps to retain them.
  - Recommending products based on customer preferences: These suggest products or services to customers based on their past purchases or browsing history.

**4. Education**
  - Predicting student performance and outcomes: It analyze factors like attendance, grades, and study habits to predict how well a student might perform in exams or courses.
  - Identifying factors affecting student dropout rates: They help schools understand why students drop out, such as financial issues or academic struggles, so they can intervene early.
  - Personalizing learning paths for students: These are recommend tailored learning materials or courses based on a student’s strengths and weaknesses.

## **Conclusion**
To summarize, in this article we learned about decision trees. On what basis the tree splits the nodes and how to can stop overfitting. why linear regression doesn’t work in the case of classification problems.To check out the full implementation of these please refer to my Github repository. You can master all the Data Science topics with our Black Belt Plus Program with out 50+ projects and 20+ tools. We hope you like this article. We aim to provide a clear understanding of the decision tree algorithm. You’ll also find decision tree examples that will help you understand the concepts better. Start your learning journey today!
