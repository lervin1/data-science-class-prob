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

# creating tree based recipe ----
airbnb_fe_tree_rec <- recipe(host_is_superhost ~review_scores_rating +
                               review_scores_cleanliness + beds + accommodates +
                               maximum_minimum_nights + host_total_listings_count +
                               host_response_time + host_identity_verified +
                               room_type, data = train_airbnb) |> 
  step_impute_mode(all_nominal_predictors()) |>
  step_impute_median(all_numeric_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# seeing if recipe works
fe_tree_recipe_check <- prep(airbnb_ks_tree_rec) |> 
  bake(new_data = NULL)

# save recipes ----
save(airbnb_fe_tree_rec, file = here("attempt_3/recipes/airbnb_fe_tree_rec.rda"))