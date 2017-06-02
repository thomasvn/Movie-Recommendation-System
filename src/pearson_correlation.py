import config
import math


def pearson_correlation_prediction():
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
        average_user_rating = sum_of_user_ratings/length_of_user_ratings

        # Iterate through all user data that requires predictions
        for user_data in config.USER_DATA_DICTIONARY[user_id]:
            if user_data[1] == 0:
                movie_id_of_to_be_predicted = user_data[0]
                relevant_training_data_info = []  # List of dictionaries. {'training_data_id': int, 'common_movie_ratings': {}, 'avg_rating': int, 'cos_sim': float}

                # Find all training data users that have also rated the movie we are trying to predict
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

                        # # Normalize the movie ratings by subtracting the average from all ratings
                        # for movie_id in commonly_rated_movies:
                        #     commonly_rated_movies[movie_id] -= average_movie_rating

                        # Calculate the cosine similarity between the user vector and the training data vector
                        similarity = 0
                        user_normalized = 0
                        training_normalized = 0
                        for movie_id in commonly_rated_movies:
                            user_difference = filtered_user_data_dictionary[movie_id] - average_user_rating
                            training_difference = commonly_rated_movies[movie_id] - average_movie_rating
                            similarity += (user_difference * training_difference)
                            user_normalized += (user_difference ** 2)
                            training_normalized += (training_normalized ** 2)

                        if len(commonly_rated_movies) == 1:
                            cos_similarity = 0.95
                        else:
                            normalize = float(math.sqrt(user_normalized) * math.sqrt(training_normalized))
                            if normalize == 0:
                                print commonly_rated_movies
                                for movie_id in commonly_rated_movies:
                                    print "training: " + str(training_data_id) + " user: " + str(user_id)
                                    print str(filtered_user_data_dictionary[movie_id]) + " " + str(commonly_rated_movies[movie_id])
                                return
                            cos_similarity = float(similarity / normalize)
                        print cos_similarity

                        single_training_data_info = {
                            'training_data_id': (training_data_id + 1),
                            'common_movie_ratings_normalized': commonly_rated_movies,
                            'avg_rating': average_movie_rating,
                            'rating_of_movie_to_be_predicted': config.TRAINING_DATA_MATRIX[training_data_id][movie_id_of_to_be_predicted-1],
                            'cos_sim': cos_similarity
                        }
                        relevant_training_data_info.append(single_training_data_info)

                print relevant_training_data_info  # Contains up to 200 testing data points for single prediction
                break

        break
