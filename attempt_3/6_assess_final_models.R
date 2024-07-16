# Classification Prediction Problem ----
# Assess Top Model w/ ks rec

# load packages
library(tidyverse)
library(tidymodels)
library(here)

# load data
load(here("attempt_3/results/final_fit_3.rda"))
load(here("data/test_airbnb.rda"))

predicted_prob_bt_2 <-  predict(final_fit_3, test_airbnb, type = "prob")

prob_result_bt_2 <- test_airbnb |>
  select(id) |>
  bind_cols(predicted_prob_bt_2) |>
  select(id, .pred_1) |>
  rename(predicted = .pred_1)

prob_result_bt_2

# making boosted tree submission
write_csv(prob_result_bt_2, file = here("submissions/bt_submission_attempt_2.csv"))