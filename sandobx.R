library(textmineR)
library(keras)
library(magrittr)

### Format data ----

x_train <- nih_sample_dtm

### Format some training parameters ----

num_hidden <- 100
num_topic <- 20
batch_size <- 100
alpha <- 1 / 20

mu1 <- log(alpha) - 1 / num_topic * num_topic * log(alpha) # this is zero
sigma1 <- 1 / alpha * (1 - 2 / num_topic) + 1 / (num_topic ^ 2) * num_topic / alpha
inv_sigma1 <- 1 / sigma1
log_det_sigma <- num_topic * log(sigma1)


### Define the network and some other stuff ----

# network here

# sampling function
sampling <- function(z_mean, z_log_var, batch_size, num_topic) {
  
  epsilon <- rnorm(n = batch_size * num_topic, 
                   mean = 0, sd = 1)
  
  epsilon <- matrix(epsilon, nrow = batch_size, ncol = num_topic)
  
  out <- z_mean + exp(z_log_var / 2) * epsilon
  
  out
}


