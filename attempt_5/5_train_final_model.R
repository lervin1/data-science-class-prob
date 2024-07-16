# Classification Prediction Problem ----
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data and rf tuned
load(here("attempt_5/results/btree_tuned_4.rda"))
load(here("data/train_airbnb.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# finalize workflow
final_wflow_5 <- btree_tuned_4 |> 
  extract_workflow(btree_tuned_4) |>  
  finalize_workflow(select_best(btree_tuned_4, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(1234)
final_fit_5 <- fit(final_wflow_5, train_airbnb)

# save out final fit
save(final_fit_5, file = here("attempt_5/results/final_fit_5.rda"))
save(final_wflow_5, file = here("attempt_5/results/final_wflow_5.rda"))