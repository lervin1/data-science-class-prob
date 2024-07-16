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

# creating tree based kitchen sink recipe ----
airbnb_ks_tree_rec <- recipe(host_is_superhost ~., data = train_airbnb) |> 
  step_rm(first_review, last_review, id) |> 
  # keeping/imputing variables on the cusp of 20% missingness
  # get rid of id and date variables
  step_impute_mode(all_nominal_predictors()) |>
  step_impute_mean(all_numeric_predictors()) |>
  # impute mean instead of median
  step_novel(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# seeing if recipe works
ks_tree_recipe_check <- prep(airbnb_ks_tree_rec) |> 
  bake(new_data = NULL)

# save recipes ----
save(airbnb_ks_tree_rec, file = here("attempt_5/recipes/airbnb_ks_tree_rec.rda"))