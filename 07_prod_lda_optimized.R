################################################################################
# This script runs the LSI analysis using SigOpt optimized parameters
################################################################################

source("00_globals.R")

### declare parameters from SigOpt optimization ----
# note: these are hard-coded because of my use of Google Cloude ML and difficulty
# copying objects back and forth. Whatevs.

alpha <- 0.9084903271440449

hidden_nodes <- 900

k <- 380

nodes <- 862

### load data ----
load("data_derived/20_newsgroups_formatted.RData")


### create model and classify train2 and test ----
set.seed(random_seed)

# create a model from train1
prod_lda <- train_prodlda(dtm = dtm[train1[1:6500], ], # because batch_size must be divisible by 100
                          k = k,
                          alpha = alpha,
                          hidden_nodes = hidden_nodes,
                          batch_size = 100,
                          validation_split = 0.2,
                          epochs = 100)


# apply it to train2
prod_lda2 <- predict_prod_lda(prod_lda$model, 
                              dtm[c(train2, train2[1:35]), ], # because batch_size must be divisible by 100
                              100,
                              norm_theta = FALSE)

prod_lda2 <- prod_lda2[seq_along(train2), ]

prod_lda2[is.na(prod_lda2) | is.infinite(prod_lda2) ] <- 0

# apply it to test
prod_lda3 <- predict_prod_lda(prod_lda$model, 
                              dtm[c(test, test[1:33]), ], # because batch_size must be divisible by 100
                              100,
                              norm_theta = FALSE)

prod_lda3 <- prod_lda3[seq_along(test), ]

prod_lda3[is.na(prod_lda3) | is.infinite(prod_lda3) ] <- 0

### train a classifier and predict for test ----

set.seed(random_seed)

m_prod <- train_classifier(y = doc_class[train2],
                           x = prod_lda2,
                           nodes = nodes,
                           dropout = 0.3,
                           activation = "relu")

p_prod <- predict_classifier(object = m_prod$net,
                             new_data = prod_lda3)

### evaluate the result ----
e_prod <- evaluate(actual = doc_class[test],
                  predicted = apply(p_prod, 1, function(x) names(x)[which.max(x)][1]))

e_class_prod <- as.data.frame(do.call(rbind, lapply(e_prod$metrics, unlist)))


### save relevant objects ----
save(prod_lda, prod_lda2, prod_lda3, m_prod, p_prod, e_prod, e_class_prod,
     file = "data_derived/prod_lda_optimized_result.RData")

