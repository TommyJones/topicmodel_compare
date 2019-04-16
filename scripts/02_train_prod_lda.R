################################################################################
# This script fits a ProdLDA model on train1 and projects train2 into that space
################################################################################

rm(list = ls())

### load libraries ----
library(keras)
library(magrittr)
library(R6) # uh oh...
library(textmineR)

### load data ----
load("data_derived/20_newsgroups_formatted.RData")

### declare function for training a model and projecting into new space ----
train_prodlda <- function(dtm, k, alpha, hidden_nodes, batch_size, validation_split, epochs) {
  
  ### Declare some initial variables ----
  
  V <- ncol(dtm)
  
  mu1 <- log(alpha) - 1 / k * k * log(alpha) # copied but not sure if this formula is correct
  sigma1 <- 1 / alpha * (1 - 2 / k) + 1 / (k ^ 2) * k / alpha
  inv_sigma1 <- 1 / sigma1
  log_det_sigma <- k * log(sigma1)
  
  ### VAE encoder network ----
  
  # fully connected encoding layers
  x <- layer_input(batch_shape = c(batch_size, V)) 
  
  h <- x %>%
    layer_dense(units = hidden_nodes, activation = "softplus") %>%
    layer_dense(units = hidden_nodes, activation = "softplus")
  
  # the input ends up being encoded into these two parameters
  z_mean <- h %>% 
    layer_dense(units = k) %>%
    layer_batch_normalization() # units = k
  
  z_log_var <- h %>%
    layer_dense(units = k) %>%
    layer_batch_normalization() # units = k
  
  
  ### Latent space sampling function ----
  sampling <- function(args) { # note that this is the same format as python...
    
    c(z_mean, z_log_var) %<-% args # assigns z_mean and z_log_var to args
    
    epsilon <- k_random_normal(shape = list(batch_size, k),
                               mean = 0, stddev = 1)
    
    z_mean + k_exp(z_log_var / 2) * epsilon
    
  }
  
  unnormalized_z <- list(z_mean, z_log_var) %>% 
    layer_lambda(f = sampling, output_shape = k)
  
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
    x = dtm, y = NULL,
    epochs = epochs, 
    batch_size = batch_size,
    validation_split = validation_split, 
    callback_early_stopping(patience = 3),
    verbose = 1
  )
  
  ### Prepare elements for extraction ----
  weights <- get_weights(prod_lda)
  
  exp_beta <- exp(weights[[ length(weights) - 5 ]])
  phi <- exp_beta / rowSums(exp_beta)
  
  colnames(phi) <- colnames(dtm)
  
  rownames(phi) <- paste0("t_", seq_len(nrow(phi)))
  
  
  intermediate_layer_model <- keras_model(inputs = prod_lda$input,
                                          outputs = get_layer(prod_lda, index = 8)$output)
  
  theta <- exp(predict(intermediate_layer_model, dtm, batch_size = batch_size))
  
  theta <- theta / rowSums(theta)
  
  colnames(theta) <- rownames(phi)
  
  rownames(theta) <- rownames(dtm)
  
  r2 <- CalcTopicModelR2(dtm = dtm, phi = phi, theta = theta)
  
  coherence <- CalcProbCoherence(phi, dtm)
  
  ### return the result
  list(model = prod_lda,
       theta = theta,
       phi = phi,
       coherence = coherence,
       r2 = r2,
       history = history,
       batch_size = batch_size,
       # weights = weights,
       alpha = alpha)
  
}

predict_prod_lda <- function(model, new_data, batch_size){
  
  intermediate_layer_model <- keras_model(inputs = model$input,
                                          outputs = get_layer(model, index = 8)$output)
  
  theta <- exp(predict(intermediate_layer_model, new_data, batch_size = batch_size))
  
  theta <- theta / rowSums(theta)
  
  rownames(theta) <- rownames(new_data)
  
  colnames(theta) <- paste0("t_", seq_len(ncol(theta)))
  
  theta
  
}

### train and project ----
prod_lda <- train_prodlda(dtm = dtm[train1[1:6500], ], # because batch_size must be divisible by 100
                          k = 200,
                          alpha = 0.1,
                          hidden_nodes = 400,
                          batch_size = 100,
                          validation_split = 0.2,
                          epochs = 100)

prod_lda2 <- predict_prod_lda(prod_lda$model, 
                              dtm[c(train2, train2[1:35]), ], # because batch_size must be divisible by 100
                              100)

prod_lda2 <- prod_lda2[seq_along(train2), ]

prod_lda3 <- predict_prod_lda(prod_lda$model, 
                              dtm[c(test, test[1:33]), ], # because batch_size must be divisible by 100
                              100)

prod_lda3 <- prod_lda3[seq_along(test), ]

### save necessary results ----

save(prod_lda, prod_lda2, prod_lda3, file = "data_derived/prod_lda.RData")

