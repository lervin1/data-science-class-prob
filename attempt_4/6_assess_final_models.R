# Classification Prediction Problem ----
# Assess Top Model w/ ks rec

# load packages
library(tidyverse)
library(tidymodels)
library(here)

# load data
load(here("attempt_4/results/final_fit_4.rda"))
load(here("data/test_airbnb.rda"))

predicted_prob_bt_3 <-  predict(final_fit_4, test_airbnb, type = "prob")

prob_result_bt_3 <- test_airbnb |>
  select(id) |>
  bind_cols(predicted_prob_bt_3) |>
  select(id, .pred_1) |>
  rename(predicted = .pred_1)

prob_result_bt_3

# making boosted tree submission
# this is technically btree attempt 4 because of a mis-save on my part
# which is why in submissions this will be called btree attempt 4 when truly
# it is attempt btree attempt 3 on the r-scripts if that makes sense
write_csv(prob_result_bt_3, file = here("submissions/bt_submission_attempt_4.csv"))