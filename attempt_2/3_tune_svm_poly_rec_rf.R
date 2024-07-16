# Classification Prediction Problem ----
# Define and fit SVM polynomial model with tree based recipe

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)
library(tictoc)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoParallel(cores = num_cores - 1)

# load resamples ----
load(here("attempt_2/data-splits/airbnb_folds.rda"))

# load preprocessing/recipe ----
load(here("attempt_2/recipes/airbnb_ks_tree_rec.rda"))

# model specifications ----
svm_poly_model <- svm_poly(
  mode = "classification", 
  cost = tune(),
  degree = tune(),
  scale_factor = tune() ) |>
  set_engine("kernlab")

# define workflows ----
svm_poly_wflow_ks_tree <- workflow() |>
  add_model(svm_poly_model) |>
  add_recipe(airbnb_ks_tree_rec)

# hyperparameter tuning values ----
svm_poly_param <- hardhat::extract_parameter_set_dials(svm_poly_model)

# svm_poly_grid <- grid_regular(svm_poly_param, levels = 5)
svm_poly_grid <- grid_latin_hypercube(svm_poly_param, size = 50)

# fit workflow/model ----
tic("SVM POLY: KS REC TREE") # start clock

# tuning code in here
tune_svm_poly_rec_ks_tree <- svm_poly_wflow_ks_tree |>
  tune_grid(
    resamples = airbnb_folds, 
    grid = svm_poly_grid,
    control = control_grid(save_workflow = TRUE), 
    metrics = metric_set(roc_auc)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_poly_rec_ks_tree <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----

save(tune_svm_poly_rec_ks_tree, tictoc_svm_poly_rec_ks_tree, 
     file = here("attempt_2/results/tune_svm_poly_tree.rda"))
