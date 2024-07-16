# Classification Prediction Problem ----
# Train final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data and rf tuned
load(here("attempt_2/results/tune_svm_poly_rec_ks.rda"))
load(here("data/train_airbnb.rda"))

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = num_cores)

# finalize workflow
final_wflow_2 <- tune_svm_poly_rec_ks |> 
  extract_workflow(tune_svm_poly_rec_ks) |>  
  finalize_workflow(select_best(tune_svm_poly_rec_ks, metric = "roc_auc"))

# train final model ----
# set seed
set.seed(1234)
final_fit_2 <- fit(final_wflow_2, train_airbnb)

# save out final fit
save(final_fit_2, file = here("attempt_2/results/final_fit_2.rda"))
save(final_wflow_2, file = here("attempt_2/results/final_wflow_2.rda"))