import cosine_similarity

TRAINING_DATA_MATRIX = []  # 200x1000 matrix of training data
USER_DATA_DICTIONARY = {}  # Dictionary of all unfiltered user data {201: [[237,4],[306,5], ...], 202: [[]] ... }


# ------------------------------------------- Setup Configurations -----------------------------------------------------
NUM_NEAREST_NEIGHBORS = 10
USER_DATA_TEST_FILE = "test5.txt"
# USER_DATA_TEST_FILE = "test10.txt"
# USER_DATA_TEST_FILE = "test20.txt"


# ---------------------------------------------- Pre-processing --------------------------------------------------------
# Parse train.txt to place into 2D matrix
with open("../data/train.txt") as openfileobject:
    for line in openfileobject:
        line = line.split("\t")  # Split text file by tabs
        raw_movie_ratings = line

        # Change all items in array of ratings to integers and remove '\n' character
        filtered_movie_ratings = []
        for movie_rating in raw_movie_ratings:
            filtered_movie_ratings.append(int(movie_rating.rstrip()))

        TRAINING_DATA_MATRIX.append(filtered_movie_ratings)


# Parse test5.txt and place into user_data_dictionary
with open("../data/" + USER_DATA_TEST_FILE) as openfileobject:
    if USER_DATA_TEST_FILE == "test5.txt":
        USER_PREDICTION_MIN_ID = 201
        USER_PREDICTION_MAX_ID = 301
    elif USER_DATA_TEST_FILE == "test10.txt":
        USER_PREDICTION_MIN_ID = 301
        USER_PREDICTION_MIN_ID = 401
    elif USER_DATA_TEST_FILE == "test20.txt":
        USER_PREDICTION_MIN_ID = 401
        USER_PREDICTION_MIN_ID = 501

    for line in openfileobject:
        line = line.replace("\r\n", "")
        line = line.split(" ")
        if not int(line[0]) in USER_DATA_DICTIONARY:
            USER_DATA_DICTIONARY[int(line[0])] = [[int(line[1]), int(line[2])]]
        else:
            current_ratings = USER_DATA_DICTIONARY[int(line[0])]
            current_ratings += [[int(line[1]), int(line[2])]]


# -------------------------------------------- Execution Configurations ------------------------------------------------
cosine_similarity.cosine_similarity_prediction()