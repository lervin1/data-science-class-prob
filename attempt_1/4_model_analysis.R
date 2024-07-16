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
load(here("attempt_1/results/btree_tuned_1.rda"))
load(here("attempt_1/results/en_tuned_1.rda"))
load(here("attempt_1/results/fit_logistic_1.rda"))
load(here("attempt_1/results/knn_tuned_1.rda"))
load(here("attempt_1/results/rf_tuned_1.rda"))

# load testing data
load(here("data/test_airbnb.rda"))

# creating table to see best models ----
# boosted tree
tbl_btree_1 <- btree_tuned_1 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Boosted Tree")

# elastic net
tbl_en_1 <- en_tuned_1 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Elastic Net")

# k-nearest neighbor
tbl_knn_1 <- knn_tuned_1 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "K-Nearest Neighbor")

# random forest
tbl_rf_1 <- rf_tuned_1 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Random Forest")

# logistic 
tbl_logistic_1 <- fit_logistic_1 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Logistic")


# creating results tables
results_table_1 <- bind_rows(tbl_logistic_1, tbl_rf_1, tbl_knn_1,
                             tbl_en_1, tbl_btree_1) |> 
  select(model, mean, std_err, n) |> 
  arrange(desc(mean)) |> 
  rename("Mean ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n) |> 
  kable()

results_table_1

# looking at the tuning parameters to see what i can change
fig_6 <- autoplot(btree_tuned_1, metric = "roc_auc")

# ggsave
ggsave(filename = "figures/fig_6.png", fig_6, width = 6, height = 4)

# save results tables
save(results_table_1, file = here("attempt_1/results/results_table_1.rda"))
