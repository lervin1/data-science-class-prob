# Classification Prediction Problem ----
# Define and tune third boosted tree model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(bonsai)

# handle common conflicts
tidymodels_prefer()

# load folds data
load(here("attempt_4/data-splits/airbnb_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("attempt_4/recipes/airbnb_ks_tree_rec.rda"))

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
  set_engine("lightgbm") |> 
  set_mode("classification")

# define workflows ----
btree_wflow_3 <- workflow() |> 
  add_model(btree_mod) |> 
  add_recipe(airbnb_ks_tree_rec)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(btree_mod)

# change hyperparameter ranges
btree_params <- hardhat::extract_parameter_set_dials(btree_mod) |> 
  update(mtry = mtry(c(1, 20)),
         learn_rate = learn_rate(range = c(0.2, 0.5)),
         trees = trees(range = c(750, 1500)))

# build tuning grid
btree_grid <- grid_regular(btree_params, levels = c(5, 3, 4, 5))

# fit workflows/models ----
# set seed
set.seed(1234)
# tune model
btree_tuned_3 <- 
  btree_wflow_3 |> 
  tune_grid(
    airbnb_folds, 
    grid = btree_grid, 
    control = control_grid(save_workflow = TRUE)
  )

# looking at results
btree_results_3 <- btree_tuned_3 |> 
  collect_metrics()

# write out results (fitted/trained workflows) ----
save(btree_wflow_3, file = here("attempt_4/results/btree_wflow_3.rda"))
save(btree_tuned_3, file = here("attempt_4/results/btree_tuned_3.rda"))
save(btree_results_3, file = here("attempt_4/results/btree_results_3.rda"))