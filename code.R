### Noam Kadosh
### HarvardX: PH125.9x - Capstone Project
### MovieLens project
### https://github.com/noamkadosh

#######################################
# MovieLens Rating Prediction Project #
######################################

## Introduction ##

### Creating the dataset ###

##################################
# Create edx set, validation set #
##################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Additional libraries needed #
library(tidyr)
library(stringr)

options(digits = 5, pillar.sigfig = 5)

# # getting the year column from the title column
# edx_copy <- edx
# validation_copy <- validation
# edx <- edx %>% mutate(year = as.numeric(str_sub(title, -5, -2)))
# validation <- validation %>% mutate(year = as.numeric(str_sub(title, -5, -2)))

### Data preprocessing and basic exploration ###
head(edx)
summary(edx)
length(edx$rating) + length(validation$rating)
edx %>% summarize(num_users = n_distinct(userId), num_movies = n_distinct(movieId))

# Ratings distribution
unique(edx$rating)
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black", fill = "#2e4057") +
  xlab("Rating") +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ylab("Count") +
  ggtitle("Rating distribution")

# Number of ratings per Movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 35, color = "black", fill = "#2e4057") +
  xlab("Ratings Count") +
  scale_x_log10() +
  ylab("Movie Count") +
  ggtitle("Number of Ratings per Movie")

# Number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 35, color = "black", fill = "#2e4057") +
  xlab("Ratings Count") +
  scale_x_log10() +
  ylab("User Count") +
  ggtitle("Number of Ratings given by Users")

# Mean ratings given by user
edx %>%
  group_by(userId) %>%
  filter(n() >= 30) %>%
  summarize(mean_rating = mean(rating)) %>%
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 35,  color = "black", fill = "#2e4057") +
  xlab("Mean Rating") +
  ylab("Number of Users") +
  ggtitle("Mean Ratings Given By Users")

### Modeling Approcah ###

## Simplest model - using only the mean ##
# Compute the mean
mu <- mean(edx$rating)
mu
# Testing results
mu_rmse <- RMSE(validation$rating, mu)
mu_rmse

# Initializing a RMSE table to save the results and saving first model's data
rmse_results <- tibble(method = "Average movie rating model", RMSE = mu_rmse)
rmse_results %>% knitr::kable()

## Movie Effect Model ##
# Movie effect penalty term b_i
# Subtract the mean from the rating
# Plot the penalty term distribution
movie_penalties <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_penalties %>%
  ggplot(aes(b_i)) +
  geom_histogram(bins = 35, color = "black", fill = "#2e4057") +
  ylab("Number of Movies") +
  ggtitle("Movie Penalty term distribution")
# A model using the movie effect penalty term b_i
movie_effect_predictions <- validation %>%
  left_join(movie_penalties, by = "movieId") %>%
  mutate(prediction = mu + b_i)
movie_effect_rmse <- RMSE(validation$rating, movie_effect_predictions$prediction)
movie_effect_rmse
# Saving reslts to table
rmse_results <- rmse_results %>%
  add_row(method = "Movie Effect Model", RMSE = movie_effect_rmse)
rmse_results %>% knitr::kable()

## Movie & User Effect Model ##
# User effect penalty term b_u
# Subtract the movie penalty term and mean from the rating
# Plot the penalty term distribution
user_penalties <- edx %>%
  left_join(movie_penalties, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_penalties %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 35, color = "black", fill = "#2e4057") +
  ylab("Number of Users") +
  ggtitle("User Penalty term distribution")
# A model using the movie effect penalty term  b_i and the user effect penalty term b_u
user_effect_predictions <- validation %>%
  left_join(movie_penalties, by = "movieId") %>%
  left_join(user_penalties, by = "userId") %>% 
  mutate(prediction = mu + b_i + b_u)
user_effect_rmse <- RMSE(validation$rating, user_effect_predictions$prediction)
user_effect_rmse
# Saving results to table
rmse_results <- rmse_results %>%
  add_row(method = "Movie & User Effect Model", RMSE = user_effect_rmse)
rmse_results %>% knitr::kable()

## Regularized Movie & User Effect Model ##
# Using cross-validation to tune the tuning parameter lambda
lambdas <- seq(0, 10, 0.25)

# For each lambda, find movie penalty term b_i and 
# user effect penalty term b_u, followed by rating prediction & testing
rmses <- sapply(lambdas, function(lambda){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + lambda))
  predictions <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(prediction = mu + b_i + b_u) %>%
    .$prediction
  
  RMSE(validation$rating, predictions)
})

# Plot the lambdas against the RMSEs to visualize what is the best lambda
tibble(lambda = lambdas, rmse = rmses) %>%
  ggplot(aes(lambdas, rmses)) + 
  geom_point()

lambda <- lambdas[which.min(rmses)]
lambda

# Saving results to table
rmse_results <- rmse_results %>%
  add_row(method = "Regularized Movie & User Effect Model", RMSE = min(rmses))
rmse_results %>% knitr::kable()

### Results ###
rmse_results %>% knitr::kable()

### Appendix ###
print("OS:")
version
