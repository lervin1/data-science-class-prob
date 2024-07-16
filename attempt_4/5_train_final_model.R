# Classification Prediction Problem ----
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(bonsai)

# handle common conflicts
tidymodels_prefer()

# load training data and rf tuned
load(here("attempt_4/results/btree_tuned_3.rda"))
load(here("data/train_airbnb.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# finalize workflow
final_wflow_4 <- btree_tuned_3 |> 
  extract_workflow(btree_tuned_3) |>  
  finalize_workflow(select_best(btree_tuned_3, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(1234)
final_fit_4 <- fit(final_wflow_4, train_airbnb)

# save out final fit
save(final_fit_4, file = here("attempt_4/results/final_fit_4.rda"))
save(final_wflow_4, file = here("attempt_4/results/final_wflow_4.rda"))