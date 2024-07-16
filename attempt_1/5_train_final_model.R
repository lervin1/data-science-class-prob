# Classification Prediction Problem ----
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data and rf tuned
load(here("attempt_1/results/btree_tuned_1.rda"))
load(here("data/train_airbnb.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# finalize workflow
final_wflow_1 <- btree_tuned_1 |> 
  extract_workflow(btree_tuned_1) |>  
  finalize_workflow(select_best(btree_tuned_1, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(1234)
final_fit_1 <- fit(final_wflow_1, train_airbnb)

# save out final fit
save(final_fit_1, file = here("attempt_1/results/final_fit_1.rda"))
save(final_wflow_1, file = here("attempt_1/results/final_wflow_1.rda"))