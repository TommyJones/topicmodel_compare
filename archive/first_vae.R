################################################################################
# This script is my attempt to implement a simple AVITM in R using Keras
# Sources of inspiration: 
#    https://github.com/nzw0301/keras-examples/blob/master/prodLDA.ipynb
#    Deep Learning In R, p 279 - 283
################################################################################

### Load libraries ----

library(keras)
library(magrittr)
library(R6) # uh oh...
library(textmineR)

### Load sample data ----

# for now a small corpus from textmineR
# x <- nih_sample_dtm

load("~/Documents/TM_R2/RsquaredTopicModels/data_derived/NIH_DTM.RData")

x <- dtm

x <- x[, colSums(x) > 1]

# x_train <- as.matrix(x)

x_train <- x

# # dataset exactly from the example
# V <- 10922
# 
# x_train <- dataset_reuters(num_words = V)
# 
# word_index <- dataset_reuters_word_index()
# 
# index2word <- word_index$items


### Declare some initial variables ----

V <- ncol(x_train)

num_hidden <- 100 # number of nodes in the hidden layer?
num_topic <- 20 # number of topics
batch_size <- 10
alpha <- 1 / 20 # symmetric alpha

mu1 <- log(alpha) - 1 / num_topic * num_topic * log(alpha) # copied but not sure if this formula is correct
sigma1 <- 1 / alpha * (1 - 2 / num_topic) + 1 / (num_topic ^ 2) * num_topic / alpha
inv_sigma1 <- 1 / sigma1
log_det_sigma <- num_topic * log(sigma1)

### VAE encoder network ----

# fully connected encoding layers
x <- layer_input(batch_shape = c(batch_size, V)) 

h <- x %>%
  layer_dense(units = num_hidden, activation = "softplus") %>%
  layer_dense(units = num_hidden, activation = "softplus")

# the input ends up being encoded into these two parameters
z_mean <- h %>% 
  layer_dense(units = num_topic) %>%
  layer_batch_normalization() # units = num_topic

z_log_var <- h %>%
  layer_dense(units = num_topic) %>%
  layer_batch_normalization() # units = num_topic


### Latent space sampling function ----
sampling <- function(args) { # note that this is the same format as python...
  
  c(z_mean, z_log_var) %<-% args # assigns z_mean and z_log_var to args
  
  epsilon <- k_random_normal(shape = list(batch_size, num_topic),
                             mean = 0, stddev = 1)
  
  z_mean + k_exp(z_log_var / 2) * epsilon
  
}

unnormalized_z <- list(z_mean, z_log_var) %>% 
  layer_lambda(f = sampling, output_shape = num_topic)

### VAE decoder network ----

theta <- unnormalized_z %>%
  layer_activation(activation = "softmax") %>%
  layer_dropout(rate = 0.5)

doc <- theta %>%
  layer_dense(units = V) %>% # DOES THIS NEED AN ACTIVATION ARGUMENT?
  layer_batch_normalization() %>%
  layer_activation("softmax")



### Custom layer used to compute the VAE loss ----

CustomVariationalLayer <- R6Class("CustomVariationalLayer",
                                  inherit = KerasLayer,
                                  public = list(
                                    vae_loss = function(x, inference_x) {
                                      
                                      decoder_loss <- k_sum(x * k_log(inference_x), axis = -1)
                                      
                                      encoder_loss <- -0.5 * 
                                        (k_sum(inv_sigma1 * k_exp(z_log_var) + k_square(z_mean) * 
                                                 inv_sigma1 - 1 - z_log_var, axis = -1) + log_det_sigma)
                                      
                                      -1 * k_mean(encoder_loss + decoder_loss)
                                    },
                                    # custom layers are implemented by writing
                                    # a 'call' method
                                    call = function(inputs, mask = NULL) {
                                      x <- inputs[[1]]
                                      
                                      inference_x <- inputs[[2]]
                                      
                                      loss <- self$vae_loss(x, inference_x)
                                      
                                      self$add_loss(loss, inputs = inputs) 
                                      
                                      # you don't use this output, but the layer 
                                      # must return something
                                      x 
                                    }
                                  ))

# wraps the R6 class in a standard keras layer function
layer_variational <- function(object) {
  create_layer(CustomVariationalLayer, object, list())
}

y <- list(x, doc) %>%
  layer_variational()


### Training the VAE ----
prod_lda <- keras_model(x, y)

prod_lda %>% compile(
  optimizer = optimizer_adam(lr = 0.001, beta_1 = 0.99),
  loss = NULL
)

history <- prod_lda %>% fit(
  x = x_train, y = NULL,
  epochs = 100, 
  batch_size = batch_size,
  validation_split = 0.1, 
  callback_early_stopping(patience = 3),
  verbose = 1
)

### Extract relevant objects ----
weights <- get_weights(prod_lda)

exp_beta <- exp(weights[[ length(weights) - 5 ]])
phi <- exp_beta / rowSums(exp_beta)

colnames(phi) <- colnames(x_train)


intermediate_layer_model <- keras_model(inputs = prod_lda$input,
                                        outputs = get_layer(prod_lda, index = 8)$output)

theta <- exp(predict(intermediate_layer_model, x_train, batch_size = 10))

theta <- theta / rowSums(theta)

r2 <- CalcTopicModelR2(dtm = x_train, phi = phi, theta = theta)

r2


### update as an lda model ----
m <- list(phi = phi, alpha = rep(1/20,20), beta = rep(0.05, ncol(x_train)))

class(m) <- "lda_topic_model"

m <- update(m, dtm = x_train, iterations = 300, burnin = 250, 
            optimize_alpha = TRUE, calc_likelihood = TRUE, calc_r2 = TRUE)

comp <- data.frame(coh_prod = round(CalcProbCoherence(phi, x_train),3),
                   coh_lda = round(m$coherence, 3),
                   terms_prod = apply(GetTopTerms(phi, 5), 2, function(x) paste(x, collapse = ", ")),
                   terms_lda = apply(GetTopTerms(m$phi, 5), 2, function(x) paste(x, collapse = ", ")),
                   stringsAsFactors = FALSE)

### Save the model (properly w/ keras/tensorflow) ----
