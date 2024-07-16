# Classification Prediction Problem ----
# Analysis of tuned and trained models 

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(knitr)

# handle common conflicts
tidymodels_prefer()

# load fit/tuned models
load(here("attempt_5/results/btree_tuned_4.rda"))

# load testing data
load(here("data/test_airbnb.rda"))

# creating table to see best models ----
# boosted tree
tbl_btree_5 <- btree_tuned_4 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err, mtry, learn_rate, trees, min_n) |> 
  mutate(model = "Best Boosted Tree Model")


# creating results tables
results_table_5 <- bind_rows(tbl_btree_5) |> 
  select(model, mean, std_err, n, learn_rate, trees, min_n, mtry) |> 
  arrange(desc(mean)) |> 
  rename("ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n,
         "Learn Rate" = learn_rate,
         "Trees" = trees)

results_table_5

# save results tables
save(results_table_5, file = here("attempt_5/results/results_table_5.rda"))
