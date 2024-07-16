# Classification Prediction Problem ----
# Analysis of tuned and trained models (comparisons)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(knitr)

# handle common conflicts
tidymodels_prefer()

# load fit/tuned models
load(here("attempt_2/results/tune_svm_poly_rec_ks.rda"))
load(here("attempt_2/results/tune_svm_poly_tree.rda"))

# load testing data
load(here("data/test_airbnb.rda"))

# creating table to see best models ----
tbl_svm_poly_ks <- tune_svm_poly_rec_ks |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "SVM Poly KS REC")

tbl_svm_poly_tree <- tune_svm_poly_rec_ks_tree |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "SVM Poly TREE REC")


# creating results tables
results_table_2 <- bind_rows(tbl_svm_poly_ks, tbl_svm_poly_tree) |> 
  select(model, mean, std_err, n) |> 
  arrange(desc(mean)) |> 
  rename("Mean ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n) |> 
  kable()

results_table_2

# save results tables
save(results_table_2, file = here("attempt_2/results/results_table_2.rda"))
