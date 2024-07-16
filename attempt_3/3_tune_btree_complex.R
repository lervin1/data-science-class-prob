# Classification Prediction Problem ----
# Define and tune boosted tree model with complex recipe

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load folds data
load(here("attempt_3/data-splits/airbnb_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("attempt_3/recipes/airbnb_fe_tree_rec.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# set seed
set.seed(1234)

# model specifications ----
btree_mod <- boost_tree(
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune(),
  trees = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("classification")

# define workflows ----
btree_wflow_2 <- workflow() |> 
  add_model(btree_mod) |> 
  add_recipe(airbnb_fe_tree_rec)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(btree_mod)

# change hyperparameter ranges
btree_params <- hardhat::extract_parameter_set_dials(btree_mod) |> 
  update(mtry = mtry(c(1, 5)),
         learn_rate = learn_rate(range = c(-5, -0.2)),
         trees = trees(range = c(50, 500)))

# build tuning grid
btree_grid <- grid_regular(btree_params, levels = c(5, 3, 4, 5))

# fit workflows/models ----
# set seed
set.seed(1234)
# tune model
btree_tuned_2 <- 
  btree_wflow_2 |> 
  tune_grid(
    airbnb_folds, 
    grid = btree_grid, 
    control = control_grid(save_workflow = TRUE)
  )

# looking at results
btree_results_2 <- btree_tuned_2 |> 
  collect_metrics()

# write out results (fitted/trained workflows) ----
save(btree_wflow_2, file = here("attempt_3/results/btree_wflow_2.rda"))
save(btree_tuned_2, file = here("attempt_3/results/btree_tuned_2.rda"))
save(btree_results_2, file = here("attempt_3/results/btree_results_2.rda"))