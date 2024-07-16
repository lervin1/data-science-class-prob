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
load(here("attempt_3/results/rf_tuned_2.rda"))
load(here("attempt_3/results/btree_tuned_2.rda"))

# load testing data
load(here("data/test_airbnb.rda"))

# creating table to see best models ----
tbl_btree <- btree_tuned_2 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err, learn_rate, trees, min_n, mtry) |> 
  mutate(model = "Feature Engineered Boosted Tree Model")

tbl_rf <- rf_tuned_2 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Random Forest Model")


# creating results tables
results_table_3 <- bind_rows(tbl_btree, tbl_rf) |> 
  select(model, mean, std_err, n) |> 
  arrange(desc(mean)) |> 
  rename("Mean ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n) |> 
  kable()

results_table_3

# creating individual result table for boosted tree
results_table_final <- bind_rows(tbl_btree) |> 
  select(model, mean, std_err, n, learn_rate, trees, min_n, mtry) |> 
  arrange(desc(mean)) |> 
  rename("ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n,
         "Learn Rate" = learn_rate,
         "Trees" = trees)

results_table_final

# save results tables
save(results_table_3, file = here("attempt_3/results/results_table_3.rda"))
save(results_table_final, file = here("attempt_3/results/results_table_final.rda"))
