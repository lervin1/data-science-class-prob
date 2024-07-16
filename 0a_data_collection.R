# Classification Prediction Problem ----
# Reading in/collection of datasets

## load packages ----

library(tidyverse)
library(here)

## load data ----

train_classification <- read_csv(here("data/train_classification.csv"),
                                 col_types = cols(id = col_character()))

test_classification <- read_csv(here("data/test_classification.csv"),
                                col_types = cols(id = col_character()))
