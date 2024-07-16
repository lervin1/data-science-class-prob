# Classification Prediction Problem ----
# Setup pre-processing/recipes

## load packages ----

library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data
load(here("data/train_airbnb.rda"))

# creating kitchen sink recipe ----
airbnb_ks_rec <- recipe(host_is_superhost ~., data = train_airbnb) |> 
  step_rm(first_review, last_review, reviews_per_month, review_scores_rating,
          review_scores_accuracy, review_scores_cleanliness, review_scores_checkin,
          review_scores_communication, review_scores_location, review_scores_value,
          host_location, id) |> 
  # getting rid of all the variables that have more than 10% missingness
  # getting rid of id and date variables
  step_impute_mode(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_zv(all_predictors()) |> 
  step_normalize(all_predictors())
  
# seeing if recipe works
ks_recipe_check <- prep(airbnb_ks_rec) |>
  bake(new_data = NULL)

# creating tree based kitchen sink recipe ----
airbnb_ks_tree_rec <- recipe(host_is_superhost ~., data = train_airbnb) |> 
  step_rm(first_review, last_review, reviews_per_month, review_scores_rating,
          review_scores_accuracy, review_scores_cleanliness, review_scores_checkin,
          review_scores_communication, review_scores_location, review_scores_value,
          host_location, id) |> 
  # getting rid of all the variables that have more than 10% missingness
  # get rid of id and date variables
  step_impute_mode(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# seeing if recipe works
ks_tree_recipe_check <- prep(airbnb_ks_tree_rec) |> 
  bake(new_data = NULL)

# save recipes ----
save(airbnb_ks_rec, file = here("attempt_1/recipes/airbnb_ks_rec.rda"))
save(airbnb_ks_tree_rec, file = here("attempt_1/recipes/airbnb_ks_tree_rec.rda"))