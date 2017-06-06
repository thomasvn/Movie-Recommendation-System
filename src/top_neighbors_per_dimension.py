import config
import math

cosine_similarity = []  # 200x100 matrix of cosine similarities


def top_neighbors_per_dimension():
# ------------------------------------- Preprocessing Training and User Data -------------------------------------------
    # Calculate the cosine similarity between every single user and the training data
    for user_id in range(config.USER_PREDICTION_MIN_ID, config.USER_PREDICTION_MAX_ID):
        filtered_user_data_dictionary = {}  # {"movie id" : "movie rating", ...}

        # Gather rating information about user
        for user_data in config.USER_DATA_DICTIONARY[user_id]:
            if user_data[1] != 0:
                filtered_user_data_dictionary[user_data[0]] = user_data[1]  # All zeros removed from user data

        # Calculate this user's similarity against 200 entries of training data
        single_user_cosine_similarity = []
        for training_data_id in range(0, 200):
            similarity = 0
            user_normalize = 0
            training_normalize = 0
            mutually_rated_movies = []

            # Find all mutually rated movies
            for movie_id in filtered_user_data_dictionary:
                training_rating = config.TRAINING_DATA_MATRIX[training_data_id][movie_id-1]
                if training_rating != 0:
                    mutually_rated_movies.append(movie_id)

            # Handle cases in which there is only 0 or 1 similar movie
            if len(mutually_rated_movies) == 0:
                single_user_cosine_similarity.append(0.01)
                continue
            elif len(mutually_rated_movies) == 1:
                # Use custom formula to calculate a pseudo cosine similarity
                training_rating = config.TRAINING_DATA_MATRIX[training_data_id][mutually_rated_movies[0]-1]
                user_rating = filtered_user_data_dictionary[mutually_rated_movies[0]]
                pseudo_cos_similarity = float((1/(abs(training_rating-user_rating)+1)))  # 1/(difference + 1)

                # Cannot let a cosine similarity reach a value of 1
                if pseudo_cos_similarity == 1:
                    pseudo_cos_similarity = 0.95

                single_user_cosine_similarity.append(pseudo_cos_similarity)
                continue

            # Calculate similarity and normalizing measurements between user and training data
            for movie_id in mutually_rated_movies:
                training_rating = config.TRAINING_DATA_MATRIX[training_data_id][movie_id-1]
                similarity += (int(training_rating) * filtered_user_data_dictionary[movie_id])
                user_normalize += (int(filtered_user_data_dictionary[movie_id]) ** 2)
                training_normalize += (int(training_rating) ** 2)

            # Normalize the cosine similarity
            normalize = float(math.sqrt(user_normalize) * math.sqrt(training_normalize))
            cos_similarity = float(similarity / normalize)

            # Append to list of user's cosine similarities against test data
            single_user_cosine_similarity.append(cos_similarity)

        # print single_user_cosine_similarity  # Should be 200 in length
        cosine_similarity.append(single_user_cosine_similarity)

    # print cosine_similarity  # Should be 100 in length


# --------------------------------------- Rating Prediction For All Users ----------------------------------------------
    f = open("../test_20.txt", "a+")
    movie_prediction_counter = [0] * 5
    for user_id in range(config.USER_PREDICTION_MIN_ID, config.USER_PREDICTION_MAX_ID):

        # Look through every single user's previous ratings and predictions that need to be completed
        for user_data in config.USER_DATA_DICTIONARY[user_id]:

            # Identify all data that needs to be predicted
            if user_data[1] == 0:
                numerator = 0
                denominator = 0
                movie_id_of_to_be_predicted = user_data[0]
                mutually_rated_movies = []
                # nearest_neighbors = []
                nearest_neighbors_k_dimension = [[0 for x in range(0)] for y in range(1001)]  # At most 1000 shared dimensions

                # Find all users that have also rated this movie
                for training_data_id in range(0, 200):
                    if config.TRAINING_DATA_MATRIX[training_data_id][movie_id_of_to_be_predicted-1] != 0:
                        mutually_rated_movies.append(training_data_id+1)

                # Calculate nearest neighbors considering all users who have rated the movie we are trying to predict
                for mutually_rated_movie_id in mutually_rated_movies:
                    mutually_rated_movie_user_cos_sim = cosine_similarity[user_id - config.USER_PREDICTION_MIN_ID][mutually_rated_movie_id - 1]
                    mutually_rated_movie_user_rating = config.TRAINING_DATA_MATRIX[mutually_rated_movie_id-1][movie_id_of_to_be_predicted-1]

                    # Do not add to list of nearest neighbors if barely even similar
                    if mutually_rated_movie_user_cos_sim < 0.05:
                        continue

                    nearest_neighbors = nearest_neighbors_k_dimension[len(mutually_rated_movies)]

                    # Maintain list of top nearest neighbors available
                    if len(nearest_neighbors) > config.NUM_NEAREST_NEIGHBORS:
                        if mutually_rated_movie_user_cos_sim > nearest_neighbors[0]['cos_sim']:
                            nearest_neighbors.pop(0)
                            nearest_neighbors.append({'id': mutually_rated_movie_id, 'cos_sim': mutually_rated_movie_user_cos_sim, 'rating': mutually_rated_movie_user_rating})
                            nearest_neighbors = sorted(nearest_neighbors, key=lambda k: k['cos_sim'])
                    else:
                        nearest_neighbors.append({'id': mutually_rated_movie_id, 'cos_sim': mutually_rated_movie_user_cos_sim, 'rating': mutually_rated_movie_user_rating})
                        nearest_neighbors = sorted(nearest_neighbors, key=lambda k: k['cos_sim'])

                # Iterate through all of nearest neighbors to calculate a weighted average
                for neighbor_dimension_k in nearest_neighbors_k_dimension:
                    for neighbor in neighbor_dimension_k:
                        numerator += (float(neighbor['cos_sim']) * neighbor['rating'])
                        denominator += float(neighbor['cos_sim'])

                # If no neighbors rated this movie, just take the average of this user's ratings
                if denominator == 0:
                    average_rating = 0
                    number_of_ratings = 0
                    for single_user_data in config.USER_DATA_DICTIONARY[user_id]:
                        if single_user_data[1] != 0:
                            average_rating += single_user_data[1]
                            number_of_ratings += 1
                    movie_prediction = average_rating/number_of_ratings
                else:
                    movie_prediction = numerator/denominator

                # Round this number either up or down
                if movie_prediction % 1 < 0.5:
                    movie_prediction = int(math.floor(movie_prediction))
                else:
                    movie_prediction = int(math.ceil(movie_prediction))

                print str(movie_prediction) + "  " + str(nearest_neighbors_k_dimension)
                movie_prediction_counter[movie_prediction-1] += 1
                f.write(str(user_id) + " " + str(movie_id_of_to_be_predicted) + " " + str(movie_prediction) + "\n")

    print "1: " + str(movie_prediction_counter[0])
    print "2: " + str(movie_prediction_counter[1])
    print "3: " + str(movie_prediction_counter[2])
    print "4: " + str(movie_prediction_counter[3])
    print "5: " + str(movie_prediction_counter[4])
