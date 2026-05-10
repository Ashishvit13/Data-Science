# **Mathematical explanation of K-Nearest Neighbour**

KNN stands for K-nearest neighbour is a popular algorithm in Supervised Learning commonly used for classification tasks. It works by classifying data based on its similarity to neighboring data points. The core idea of KNN is straightforward when a new data point is introduced the algorithm finds its K nearest neighbors and assigns the most frequent class from these neighbors to the new point.

## **Working of K-Nearest Neighbour**
KNN algorithm stores all available cases and classifies new data based on the majority class of its nearest neighbors. Value of K in KNN refers to the number of nearest neighbors to consider when performing classification.

**K** parameter is critical because:
  - If K is too small, the model may be sensitive to noise in the dataset.
  - If K is too large, the classification might be too generalized, and nuances in the data may be overlooked.

Distance between data points is measured using a distance metric, such as **Euclidean distance**, to find the nearest neighbors.

## **How do we choose K?**
Choosing the right value for K is crucial:

  - A commonly used rule of thumb is to select K ≈ sqrt(n), where n is the number of data points in the dataset.
  - If n is even adjust K to be odd by adding or subtracting 1 to avoid ties in majority voting.

Let’s dive deeper into an example of KNN to make the concept clearer. Below is a data that includes **age, gender and the class of sports** people play.

<img width="276" height="422" alt="image" src="https://github.com/user-attachments/assets/057a8d2a-3103-4ae3-833e-1e0f90776b5a" />

<img width="526" height="424" alt="image" src="https://github.com/user-attachments/assets/3b262398-78ee-4c09-9d6a-338249743136" />

<img width="518" height="440" alt="image" src="https://github.com/user-attachments/assets/780843b6-cae6-45aa-8a6f-eb4a4abc5b72" />

<img width="454" height="156" alt="image" src="https://github.com/user-attachments/assets/17e2d6e5-ab92-466c-bf2b-f002041b53b7" />

<img width="257" height="423" alt="image" src="https://github.com/user-attachments/assets/40fca25c-3f48-4fe9-a646-982672f8ccc1" />

<img width="544" height="251" alt="image" src="https://github.com/user-attachments/assets/9dee81bd-ca77-41c1-9850-e8b94ed9ef8f" />
