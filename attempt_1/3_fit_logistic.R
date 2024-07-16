# Classification Prediction Problem ----
# Define and fit first logistic model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data
load(here("attempt_1/data-splits/airbnb_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("attempt_1/recipes/airbnb_ks_rec.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# model specifications ----
logistic_spec <- 
  logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification") 

# define workflows ----
logistic_wflow_1 <- workflow() |>
  add_model(logistic_spec) |>
  add_recipe(airbnb_ks_rec)

# fit workflows/models ----
fit_logistic_1 <- fit_resamples(logistic_wflow_1, resamples = airbnb_folds, 
                              control = control_resamples(save_pred = TRUE))


# view results
logistic_results_1 <- fit_logistic_1 |> collect_metrics()

# write out results (fitted/trained workflows) ----
save(logistic_results_1, file = here("attempt_1/results/logistic_results_1.rda"))
save(logistic_wflow_1, file = here("attempt_1/results/logistic_wflow_1.rda"))
save(fit_logistic_1, file = here("attempt_1/results/fit_logistic_1.rda"))