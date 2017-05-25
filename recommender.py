import math

training_data_matrix = []
user_data_dictionary = {}
cosine_similarity = []


# Parse train.txt to place into 2D matrix
with open("train.txt") as openfileobject:
    for line in openfileobject:
        line = line.split("\t")
        training_data_matrix.append(line)


# Parse test5.txt and place into user_data_dictionary
with open("test5.txt") as openfileobject:
    for line in openfileobject:
        line = line.replace("\r\n", "")
        line = line.split(" ")
        if not int(line[0]) in user_data_dictionary:
            user_data_dictionary[int(line[0])] = [[int(line[1]), int(line[2])]]
        else:
            current_ratings = user_data_dictionary[int(line[0])]
            current_ratings += [[int(line[1]), int(line[2])]]

# Calculate the cosine similarity between every single user and the training data
for user_id in range(201, 301):
    filtered_user_data_dictionary = {}  # {"movie id" : "movie rating", ...}

    # Gather rating information about user
    for user_data in user_data_dictionary[user_id]:
        if user_data[1] != 0:
            filtered_user_data_dictionary[user_data[0]] = user_data[1]

    # Calculate user's similarity against 200 entries of training data
    similarity = 0
    normalize = 0
    user_normalize = 0
    training_normalize = 0
    mutually_rated_movies = []
    single_user_cosine_similarity = []
    for training_data_id in range(0, 200):
        # Find all mutually rated movies
        for movie_id in filtered_user_data_dictionary:
            training_rating = training_data_matrix[training_data_id][movie_id-1]
            if training_rating != "0":
                mutually_rated_movies.append(movie_id)

        # Handle cases in which there is only 0 or 1 similar movie
        if len(mutually_rated_movies) == 0:
            cos_similarity = 0
            single_user_cosine_similarity.append(cos_similarity)
            # print cosine_similarity
            continue  # TODO: Set Cosine Similarity to 0
        elif len(mutually_rated_movies) == 1:
            cos_similarity = 0.98
            single_user_cosine_similarity.append(cos_similarity)
            # print cosine_similarity
            continue  # TODO: Set Cosine Similarity to 0.98 and rating should be a 4

        # Calculate similarity and normalizing measurements between user and training data
        for movie_id in mutually_rated_movies:
            training_rating = training_data_matrix[training_data_id][movie_id - 1]
            similarity += (int(training_rating) * filtered_user_data_dictionary[movie_id])
            # print mutually_rated_movies, str(filtered_user_data_dictionary[movie_id]) + "*" + str(training_rating) + "=" + str(similarity)

            user_normalize += (int(filtered_user_data_dictionary[movie_id]) ** 2)
            training_normalize += (int(training_rating) ** 2)

        # Normalize the cosine similarity
        normalize = math.sqrt(user_normalize) * math.sqrt(training_normalize)
        cos_similarity = similarity / normalize

        single_user_cosine_similarity.append(cos_similarity)

    print single_user_cosine_similarity  # Should be 200 in length
    cosine_similarity.append(single_user_cosine_similarity)

print len(cosine_similarity)  # Should be 100 in length
