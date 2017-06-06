import config
import math

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

    # Calculation of Pearson Coefficient for every user that needs a rating prediction
    for user_id in range(config.USER_PREDICTION_MIN_ID, config.USER_PREDICTION_MAX_ID):
        filtered_user_data_dictionary = {}  # {"movie id" : "movie rating", ...}

        # Gather rating information about user
        length_of_user_ratings = 0
        sum_of_user_ratings = 0
        for user_data in config.USER_DATA_DICTIONARY[user_id]:
            if user_data[1] != 0:
                filtered_user_data_dictionary[user_data[0]] = user_data[1]
                sum_of_user_ratings += user_data[1]
                length_of_user_ratings += 1
        average_user_rating = float(sum_of_user_ratings/length_of_user_ratings)

        # Iterate through all user data that requires predictions
        for user_data in config.USER_DATA_DICTIONARY[user_id]:
            if user_data[1] == 0:
                movie_id_of_to_be_predicted = user_data[0]
                ratings_of_movie_to_be_predicted = []  # List of dictionaries {'user_id': 'rating'}
                relevant_movie_data_info = []  # List of dictionaries. Seen below in "single_training_data_info".

                # Find current ratings of movie to be predicted
                for user_id in range(0,200):
                    user_rating = TRANSPOSED_TRAINING_DATA_MATRIX[movie_id_of_to_be_predicted-1][user_id]
                    if user_rating != 0:
                        single_rating = {user_id+1: user_rating}
                        ratings_of_movie_to_be_predicted.append(single_rating)

                # Find the similarity of the movie to be predicted against all other movies
                for movie_id in range(0, 1000):
                    common_user_ratings = {}  # {'user_id': 'movie_rating', ... }

                    # Identify users who have co-rated the movie
                    average_movie_rating = 0
                    counter = 0
                    for user_id in range(0,200):
                        movie_rating = TRANSPOSED_TRAINING_DATA_MATRIX[movie_id][user_id]
                        if movie_rating != 0 and user_id+1 in ratings_of_movie_to_be_predicted:
                            common_user_ratings[user_id+1] = movie_rating
                            average_movie_rating += movie_rating
                            counter += 1

                    # If the training data user doesn't have any similarly rated movies, do not consider
                    if not bool(common_user_ratings):
                        continue
                    else:
                        average_movie_rating /= counter

                    # Calculate the similarity between the movie to be predicted vector and the training data vector
                    similarity = 0
                    user_normalized = 0
                    training_normalized = 0
                    for movie_id in common_user_ratings:
                        user_difference = ratings_of_movie_to_be_predicted[movie_id] - average_user_rating
                        training_difference = common_user_ratings[movie_id] - average_movie_rating
                        similarity += (user_difference * training_difference)
                        user_normalized += (user_difference ** 2)
                        training_normalized += (training_difference ** 2)

                    normalize = float(math.sqrt(user_normalized) * math.sqrt(training_normalized))
                    if len(common_user_ratings) == 1:
                        for movie_id in common_user_ratings:
                            difference = abs(common_user_ratings[movie_id] - ratings_of_movie_to_be_predicted[movie_id])
                            cos_similarity = float(1 / (difference + 1))
                        if cos_similarity == 1:
                            cos_similarity = 0.95
                    elif normalize == 0:
                        cos_similarity = float(0)
                    else:
                        cos_similarity = float(similarity / normalize)
                        if cos_similarity > 1:
                            cos_similarity = 1

                    single_training_data_info = {
                        'training_data_id': (training_data_id + 1),
                        'common_movie_ratings': common_user_ratings,
                        'avg_rating': average_movie_rating,
                        'rating_of_movie_to_be_predicted': config.TRAINING_DATA_MATRIX[training_data_id][movie_id_of_to_be_predicted-1],
                        'pearson_coefficient': cos_similarity
                    }
                    relevant_movie_data_info.append(single_training_data_info)

                # Calculate nearest neighbors considering all users who have rated the movie we are trying to predict
                nearest_neighbors = []
                for training_data in relevant_movie_data_info:
                    # Do not add to list of nearest neighbors if barely even similar
                    if training_data['pearson_coefficient'] < 0.05:
                        continue

                    # Maintain list of top nearest neighbors available
                    if len(nearest_neighbors) > config.NUM_NEAREST_NEIGHBORS:
                        if training_data['pearson_coefficient'] > nearest_neighbors[0]['pearson_coefficient']:
                            nearest_neighbors.pop(0)
                            nearest_neighbors.append(training_data)
                            nearest_neighbors = sorted(nearest_neighbors, key=lambda k: k['pearson_coefficient'])
                    else:
                        nearest_neighbors.append(training_data)
                        nearest_neighbors = sorted(nearest_neighbors, key=lambda k: k['pearson_coefficient'])

                # Calculate the Pearson Correlation Rating using all nearest neighbors
                numerator = 0
                denominator = 0
                for training_data in nearest_neighbors:
                    numerator += ((training_data['rating_of_movie_to_be_predicted'] - training_data['avg_rating']) * training_data['pearson_coefficient'])
                    denominator += abs(training_data['pearson_coefficient'])

                # If no other user is correlated to the user we're trying to predict, cold start
                if denominator == 0:
                    pearson_correlation_rating = average_user_rating
                else:
                    pearson_correlation_rating = average_user_rating + float(numerator/denominator)

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

                f.write(str(user_id) + " " + str(movie_id_of_to_be_predicted) + " " + str(pearson_correlation_rating) + "\n")
                print str(pearson_correlation_rating) + " " + str(relevant_movie_data_info)
                movie_prediction_counter[pearson_correlation_rating - 1] += 1

    print "1: " + str(movie_prediction_counter[0])
    print "2: " + str(movie_prediction_counter[1])
    print "3: " + str(movie_prediction_counter[2])
    print "4: " + str(movie_prediction_counter[3])
    print "5: " + str(movie_prediction_counter[4])
