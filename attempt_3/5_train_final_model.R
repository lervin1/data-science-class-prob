# Classification Prediction Problem ----
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data and rf tuned
load(here("attempt_3/results/btree_tuned_2.rda"))
load(here("data/train_airbnb.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# finalize workflow
final_wflow_3 <- btree_tuned_2 |> 
  extract_workflow(btree_tuned_2) |>  
  finalize_workflow(select_best(btree_tuned_2, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(1234)
final_fit_3 <- fit(final_wflow_3, train_airbnb)

# save out final fit
save(final_fit_3, file = here("attempt_3/results/final_fit_3.rda"))
save(final_wflow_3, file = here("attempt_3/results/final_wflow_3.rda"))