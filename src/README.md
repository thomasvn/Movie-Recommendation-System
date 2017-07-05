## Getting Started
After cloning this repository, the `main.py` and `config.py` files will need to be configured before running the recommendation system.

In `main.py` you can configure the:
* Algorithm which you want to be run

In `config.py` you can configure the:
* Number of nearest neighbors (explained later)
* Number of nearest neighbors in each dimension (used for custom algorithm)
* Training data file

## About the Algorithms
##### K-Nearest Neighbors
In collaborative filtering techniques, we compare the similarity between two different items. Either the similarity between two users, or the similarity between two objects.

After determining the similarity between one object and the rest of the given objects, we only want to consider the **most similar items**. Therefore, we will extract K number of most similar objects. These are called the K-Nearest Neighbors.

##### Cosine Similarity
A cosine similarity is the angle found between two vectors - both of which represent the objects that we are comparing.

In my code, I calculated the cosine similarity between all users which needed movies to be predicted and all users within the training data set. I then performed a weighted average of the K-Nearest Neighbors.

##### Pearson Coefficient
One factor that the cosine similarity does not take into account for is every user’s deviation of what an “average” movie should have been rated. For example, some users may rate very highly for all moves, while others may rate very lowly for all movies simply because their standards are different. The Pearson Correlation algorithm takes into account this deviation.

In our code, this meant that while we calculate for the weight or similarity of every user to user, we needed to subtract the average user rating from the actual user rating both in the numerator and denominator.

##### Inverse User Frequency (IUF) & Case Amplification
IUF is a weight that can be applied to determine the importance of each movie. The fundamental idea behind IUF is that we should accentuate the similarly rated movies which are more obscure and rare, and penalize similarly rated movies which are very common and less revealing about the user’s actual similarity.

The idea of case amplification is to increase the spread of similarity values by raising the similarity to a power of 2.5.

##### Adjusted Cosine Similarity
The Adjusted Cosine Similarity algorithm is similar to the Pearson Correlation algorithm in that we are attempting to factor in the deviation that may be found. The only difference is that in the adjusted cosine similarity we are always subtracting by the user’s average ratings as opposed to the movie’s average rating.

The Adjusted Cosine Similarity was implemented as an Item-Based Collaborative Filtering Algorithm as opposed to the User-Based algorithms we were previously working on.

##### Top-K of Each Dimension (Custom Algorithm)
Instead of finding only the top k nearest neighbors solely based on similarity coefficients, I decided to take the top k nearest neighbors for every dimension.

This acted as a throttle to having too many similar users who shared few dimensions and actually gave a chance to the users who shared many common dimensions but just had a slightly lower cosine similarity.

## Example
```
$ python main.py
```


