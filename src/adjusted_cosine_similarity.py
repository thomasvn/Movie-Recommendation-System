import config
import math

adjusted_cosine_similarities = [[0 for x in range(1000)] for y in range(1000)]  # 1000x1000 matrix of movie to movie similarities
TRANSPOSED_TRAINING_DATA_MATRIX = []  # 1000x200 matrix of training data


# ---------------------------------------- Item Based Collaborative Filtering ------------------------------------------
def adjusted_cosine_similarity():
    f = open("../test_5.txt", "a+")
    movie_prediction_counter = [0] * 5

    # Transpose the TRAINING_DATA_MATRIX of movie ratings
    for movie_id in range(0, 1000):
        single_movie_ratings = []
        for training_data_id in range(0, 200):
            single_movie_ratings.append(config.TRAINING_DATA_MATRIX[training_data_id][movie_id])
            TRANSPOSED_TRAINING_DATA_MATRIX.append(single_movie_ratings)

    # Calculate user averages for all 200 users in training data
    training_data_rating_averages = []
    for ratings_of_single_user in config.TRAINING_DATA_MATRIX:
        rating_average = 0
        count = 0
        for rating in ratings_of_single_user:
            if rating != 0:
                rating_average += 1
                count += 1
        rating_average /= count
        training_data_rating_averages.append(rating_average)

    # Calculate movie to movie similarities for all movies
    for movie_id1 in range(0,1000):

        # Find movies that have been rated by movie_id1
        movie_ratings_of_movieid1 = {}
        for user_id in range(0, 200):
            user_rating = TRANSPOSED_TRAINING_DATA_MATRIX[movie_id1][user_id]
            if user_rating != 0:
                movie_ratings_of_movieid1[user_id+1] = user_rating

        for movie_id2 in range(0,1000):
            if movie_id1 == movie_id2:
                adjusted_cosine_similarities[movie_id1][movie_id2] = 1
                continue

            # Find all users that have rated both movie_id1 as well as movie_id2 (common dimensions)
            common_user_ratings = {}  # {'user_id': 'movie_rating', ... }
            for user_id in range(0,200):
                user_rating = TRANSPOSED_TRAINING_DATA_MATRIX[movie_id2][user_id]
                if user_rating != 0 and user_id+1 in movie_ratings_of_movieid1:
                    common_user_ratings[user_id+1] = user_rating

            movie_ratings_of_movieid2 = common_user_ratings

            # Calculate the cosine similarity between the user vector and the training data vector
            similarity = 0
            rating_normalized1 = 0
            rating_normalized2 = 0
            for common_user_id in common_user_ratings:
                rating_difference1 = movie_ratings_of_movieid1[common_user_id] - training_data_rating_averages[common_user_id - 1]
                rating_difference2 = movie_ratings_of_movieid2[common_user_id] - training_data_rating_averages[common_user_id - 1]
                similarity += (rating_difference1 * rating_difference2)
                rating_normalized1 += (rating_difference1 ** 2)
                rating_normalized2 += (rating_difference2 ** 2)

            normalize = float(math.sqrt(rating_normalized1) * math.sqrt(rating_normalized2))
            if len(common_user_ratings) == 1:
                for common_user_id in common_user_ratings:
                    difference = abs(movie_ratings_of_movieid1[common_user_id] - movie_ratings_of_movieid2[common_user_id])
                    cos_similarity = float(1 / (difference + 1))
                if cos_similarity == 1:
                    cos_similarity = 0.95
            elif normalize == 0:
                cos_similarity = float(0)
            else:
                cos_similarity = float(similarity / normalize)
                if cos_similarity > 1:
                    cos_similarity = 1

            adjusted_cosine_similarities[movie_id1][movie_id2] = cos_similarity
            print cos_similarity
        return

# ---------------------------------------- Adjusted Cosine Similarity Rating -------------------------------------------
    for user_id in range(config.USER_PREDICTION_MIN_ID, config.USER_PREDICTION_MAX_ID):
        for user_data in config.USER_DATA_DICTIONARY[user_id]:
            if user_data[1] == 0:
                numerator = 0
                denominator = 0
                movie_id_of_to_be_predicted = user_data[0]

                # Iterate through all commonly rated movies
                average_user_rating = 0
                count = 0
                for user_data_rated in config.USER_DATA_DICTIONARY[user_id]:
                    if user_data_rated[1] != 0:
                        numerator += (adjusted_cosine_similarities[movie_id_of_to_be_predicted-1][user_data[0]-1] * user_data_rated[1])
                        denominator += adjusted_cosine_similarities[movie_id_of_to_be_predicted-1][user_data[0]-1]
                        average_user_rating += user_data_rated[1]
                        count += 1
                average_user_rating /= count

                # If no other user is correlated to the user we're trying to predict, cold start
                if denominator == 0:
                    pearson_correlation_rating = average_user_rating
                else:
                    pearson_correlation_rating = average_user_rating + float(numerator / denominator)

                # Round this number either up or down
                if pearson_correlation_rating % 1 < 0.5:
                    pearson_correlation_rating = int(math.floor(pearson_correlation_rating))
                else:
                    pearson_correlation_rating = int(math.ceil(pearson_correlation_rating))

                # Make sure the rating is still in bounds
                if pearson_correlation_rating > 5:
                    pearson_correlation_rating = 5
                elif pearson_correlation_rating < 1:
                    pearson_correlation_rating = 1

                f.write(str(user_id) + " " + str(movie_id_of_to_be_predicted) + " " + str(
                    pearson_correlation_rating) + "\n")
                print str(pearson_correlation_rating)
                movie_prediction_counter[pearson_correlation_rating - 1] += 1

    print "1: " + str(movie_prediction_counter[0])
    print "2: " + str(movie_prediction_counter[1])
    print "3: " + str(movie_prediction_counter[2])
    print "4: " + str(movie_prediction_counter[3])
    print "5: " + str(movie_prediction_counter[4])
