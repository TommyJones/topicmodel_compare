################################################################################
# This script is my attempt to learn VAEs using the Keras book for R
################################################################################

library(keras)
library(magrittr)

### Basic VAE Structure ----

## Encode input into a mean and variance
# c(z_mean, z_log_var) %<-% encoder(input_img)

## draw latent point using small random epsilon
# z <- z_mean + exp(z_log_var) * epsilon

## decode z back to an image
# reconstructed_img <- decoder(z)

## instantiates an autoencoder which maps an input image to its reconstruction
# model <- keras_model(input_img, reconstructed_img)

################################################################################
# code from the book

### VAE encoder network ----

img_shape <- c(28, 28, 1)

batch_size <- 16

latent_dim <- 2L # dimensionality of the latent space, a 2D plane

input_img <- layer_input(shape = img_shape)

# this is just prepping the input
x <- input_img %>%
  layer_conv_2d(filters = 32, kernel_size = 3, padding = "same",
                activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                activation = "relu", strides = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                activation = "relu") %>%
  layer_conv_2d(filters = 64, kernel_size = 3, padding = "same",
                activation = "relu")

shape_before_flattening <- k_int_shape(x)

# this is the fully connected layer
x <- x %>% 
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu")

# the input image ends up being encoded into these two parameters
z_mean <- x %>% 
  layer_dense(units = latent_dim)

z_log_var <- x %>%
  layer_dense(units = latent_dim)

### Latent space sampling function ----

sampling <- function(args) { # note that this is the same format as python...
  
  c(z_mean, z_log_var) %<-% args # assigns z_mean and z_log_var to args
  
  epsilon <- k_random_normal(shape = list(k_shape(z_mean)[1], latent_dim),
                             mean = 0, stddev = 1)
  
  z_mean + k_exp(z_log_var) * epsilon
  
}

z <- list(z_mean, z_log_var) %>% 
  layer_lambda(sampling)

### VAE decoder network ----

# input where you'll feed 'z'
decoder_input <- layer_input(k_int_shape(z)[-1])

x <- decoder_input %>%
  # upsamples the input
  layer_dense(units = prod(as.integer(shape_before_flattening[-1])), 
              activation = "relu") %>%
  # reshapes 'z' into a feature map of the same shape as the feature map just 
  # before the last layer_flatten in the encoder model
  layer_reshape(target_shape = shape_before_flattening[-1]) %>% 
  # uses a layer_conv_2d_transpose and layer_conv_2d to decode z into a feature 
  # map the same size as the original image input
  layer_conv_2d_transpose(filters = 32, kernel_size = 3, padding = "same",
                          activation = "relu", strides = c(2,2)) %>%
  # reshapes 'z' into a feature map of the same shape as the feature map just
  # before the last layer_flatten in the encoder model
  layer_conv_2d(filters = 1, kernel_size = 3, padding = "same",
                activation = "sigmoid")

# instantiates the decoder model which turns "decoder_input" into the decoded image
decoder <- keras_model(decoder_input, x) 

# applize it to 'z' to recover the decode value
z_decoded <- decoder(z)
  
### Custom layer used to compute the VAE loss ----
library(R6) # uh oh...

CustomVariationalLayer <- R6Class("CustomVariationalLayer",
                                  inherit = KerasLayer,
                                  public = list(
                                    vae_loss = function(x, z_decoded) {
                                      
                                      x <- k_flatten(x)
                                      z_decoded <- k_flatten(z_decoded)
                                      xent_loss <- metric_binary_crossentropy(x, z_decoded)
                                      kl_loss <- -5e-4 * k_mean(
                                        1 + z_log_var - k_square(z_mean) - k_exp(z_log_var),
                                        axis = -1L)
                                      
                                      k_mean(xent_loss + kl_loss)
                                    },
                                    # custom layers are implemented by writing
                                    # a 'call' method
                                    call = function(inputs, mask = NULL) {
                                      x <- inputs[[1]]
                                      z_decoded <- inputs[[2]]
                                      loss <- self$vae_loss(x, z_decoded)
                                      self$add_loss(loss, inputs = inputs) # maybe wrong?
                                      x # you don't use this output, but the layer must return something
                                    }
                                  ))

# wraps the R6 class in a standard keras layer function
layer_variational <- function(object) {
  create_layer(CustomVariationalLayer, object, list())
}

y <- list(input_img, z_decoded) %>%
  layer_variational()

### Training the VAE ----

vae <- keras_model(input_img, y)

vae %>% compile(
  optimizer = "rmsprop", # consider switching this to ADAM or adamax
  loss = NULL
)

mnist <- dataset_mnist()

c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

x_train <- x_train / 255

x_train <- array_reshape(x_train, dim = c(dim(x_train), 1))

x_test <- x_test / 255

x_test <- array_reshape(x_test, dim = c(dim(x_test), 1))

history <- vae %>% fit(
  x = x_train, y = NULL,
  epochs = 2, # should be 10
  batch_size = batch_size,
  validation_data = list(x_test, NULL)
)

