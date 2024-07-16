# Classification Prediction Problem ----
# Data cleaning/wrangling for final project

# load packages
library(tidyverse)
library(tidymodels)
library(naniar)
library(here)

# load in training data ----
load(here("data/train_airbnb.rda"))

# DATA EXPLORATION ----
# creating bar plot for target variable using training dataset
fig_1 <- train_airbnb |> 
  ggplot(aes(x = host_is_superhost)) +
  geom_bar(fill = "slategray2", color = "black") +
  theme_light() +
  labs(x = "Host is a Superhost",
       y = "Count")

# ggsave
ggsave(filename = "figures/fig_1.png", fig_1, width = 6, height = 4)

# looking at numeric variables
# correlation matrix
train_airbnb |> 
  select_if(is.numeric) |> 
  na.omit() |> 
  cor() |> 
  corrplot::corrplot(method = 'color', diag = TRUE, 
                     type = "lower", tl.cex = 0.4, 
                     tl.col ="black")

# will do step_interact with...
# review_scores_rating & review_scores_cleanliness
# beds & accommodates 
# maximum_minimum_nights & host_total_listings_count

# looking at these categorical variables...
# host response time
# host_identity verified
# room_type

# looking at host response time
fig_3 <- train_airbnb |> 
  na.omit() |> 
  ggplot(aes(x = host_is_superhost)) + 
  geom_bar(fill = "slategray2", color = "black") +
  facet_wrap(~host_response_time) +
  theme_light() +
  labs(x = "Superhost Host", y = "Count")

# ggsave
ggsave(filename = "figures/fig_3.png", fig_3, width = 6, height = 4)

# host identity verified
fig_4 <- train_airbnb |>
  ggplot(aes(x = host_is_superhost)) + 
  geom_bar(fill = "slategray2", color = "black") +
  facet_wrap(~host_identity_verified) +
  theme_light() +
  labs(x = "Superhost Host", y = "Count")

# ggsave
ggsave(filename = "figures/fig_4.png", fig_4, width = 6, height = 4)

# room type
fig_5 <- train_airbnb |>
  ggplot(aes(x = host_is_superhost)) + 
  geom_bar(fill = "slategray2", color = "black") +
  facet_wrap(~room_type) +
  theme_light() +
  labs(x = "Superhost Host", y = "Count")

# ggsave
ggsave(filename = "figures/fig_5.png", fig_5, width = 6, height = 4)

