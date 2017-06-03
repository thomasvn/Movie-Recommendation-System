import config
import math


def pearson_correlation_prediction():
    movie_prediction_counter = [0] * 5
    # Calculation of Pearson Coefficient for every user
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
                relevant_training_data_info = []  # List of dictionaries. Seen below in "single_training_data_info".

                # Find information about all training data users that have also rated the movie we are trying to predict
                for training_data_id in range(0, 200):
                    if config.TRAINING_DATA_MATRIX[training_data_id][movie_id_of_to_be_predicted - 1] != 0:
                        commonly_rated_movies = {}  # {'movie_id' : 'movie_rating', ...}

                        # Average of all of the test data's movie ratings and find commonly rated movies
                        length_of_list = 0
                        sum_of_ratings = 0
                        for movie_id in range(0, 1000):
                            movie_rating = config.TRAINING_DATA_MATRIX[training_data_id][movie_id]
                            if movie_rating != 0:
                                length_of_list += 1
                                sum_of_ratings += movie_rating
                                if (movie_id+1) in filtered_user_data_dictionary:
                                    commonly_rated_movies[movie_id+1] = movie_rating
                        average_movie_rating = sum_of_ratings/length_of_list

                        # If the training data user doesn't have any similarly rated movies, do not consider
                        if not bool(commonly_rated_movies):
                            continue

                        # Calculate the cosine similarity between the user vector and the training data vector
                        similarity = 0
                        user_normalized = 0
                        training_normalized = 0
                        for movie_id in commonly_rated_movies:
                            user_difference = filtered_user_data_dictionary[movie_id] - average_user_rating
                            training_difference = commonly_rated_movies[movie_id] - average_movie_rating
                            similarity += (user_difference * training_difference)
                            user_normalized += (user_difference ** 2)
                            training_normalized += (training_difference ** 2)

                        # TODO: why is cos_similarity 0 of user or training data ends up being all average?
                        # TODO: len=1 and normalize=0 treats the same case of different dimensions differently
                        normalize = float(math.sqrt(user_normalized) * math.sqrt(training_normalized))
                        if len(commonly_rated_movies) == 1:
                            for movie_id in commonly_rated_movies:
                                difference = abs(commonly_rated_movies[movie_id] - filtered_user_data_dictionary[movie_id])
                                cos_similarity = float(1 / (difference + 1))
                            if cos_similarity == 1:
                                cos_similarity = 0.95
                        elif normalize == 0:
                            cos_similarity = float(0)
                        else:
                            cos_similarity = float(similarity / normalize)

                        single_training_data_info = {
                            'training_data_id': (training_data_id + 1),
                            'common_movie_ratings': commonly_rated_movies,
                            'avg_rating': average_movie_rating,
                            'rating_of_movie_to_be_predicted': config.TRAINING_DATA_MATRIX[training_data_id][movie_id_of_to_be_predicted-1],
                            'pearson_coefficient': cos_similarity
                        }
                        relevant_training_data_info.append(single_training_data_info)

                # Calculate the Pearson Correlation Rating using all relevant training data
                numerator = 0
                denominator = 0
                for training_data in relevant_training_data_info:
                    numerator += ((training_data['rating_of_movie_to_be_predicted'] - training_data['avg_rating']) * training_data['pearson_coefficient'])
                    denominator += training_data['pearson_coefficient']

                pearson_correlation_rating = average_user_rating + float(numerator/denominator)

                # TODO: Pearson Correlation Rating ends up being higher than 5
                # Round this number either up or down
                if pearson_correlation_rating % 1 < 0.5:
                    pearson_correlation_rating = int(math.floor(pearson_correlation_rating))
                else:
                    pearson_correlation_rating = int(math.ceil(pearson_correlation_rating))

                print str(pearson_correlation_rating) + " " + str(relevant_training_data_info)
                movie_prediction_counter[pearson_correlation_rating - 1] += 1

    print "1: " + str(movie_prediction_counter[0])
    print "2: " + str(movie_prediction_counter[1])
    print "3: " + str(movie_prediction_counter[2])
    print "4: " + str(movie_prediction_counter[3])
    print "5: " + str(movie_prediction_counter[4])
