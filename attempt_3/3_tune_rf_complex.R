# Classification Prediction Problem ----
# Define and tune random forest model with complex recipe

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

# model specifications ----
rf_mod <-
  rand_forest(trees = 1000,
              min_n = tune(),
              mtry = tune()) |> 
  set_engine("ranger", importance = "impurity") |> 
  set_mode("classification")

# define workflows ----
# using basic tree recipe
rf_wflow_2 <- workflow() |> 
  add_model(rf_mod) |> 
  add_recipe(airbnb_fe_tree_rec)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(rf_mod)

# change hyperparameter ranges
rf_params <- parameters(rf_mod) |> 
  update(mtry = mtry(c(1, 5))) 

# build tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# fit workflows/models ----
# set seed
set.seed(1234)
# tune model
rf_tuned_2 <- 
  rf_wflow_2 |> 
  tune_grid(
    airbnb_folds, 
    grid = rf_grid, 
    control = control_grid(save_workflow = TRUE)
  )

# looking at results
rf_results_2 <- rf_tuned_2 |> 
  collect_metrics()

# write out results (fitted/trained workflows) ----
save(rf_wflow_2, file = here("attempt_3/results/rf_wflow_2.rda"))
save(rf_tuned_2, file = here("attempt_3/results/rf_tuned_2.rda"))
save(rf_results_2, file = here("attempt_3/results/rf_results_2.rda"))