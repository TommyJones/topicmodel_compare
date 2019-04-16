################################################################################
# This script trains a classifier for the 20 groups
################################################################################

rm(list = ls())

### load libraries ----
# library(randomForest)
library(keras)
library(magrittr)

### load data ----
load("data_derived/20_newsgroups_formatted.RData")
load("data_derived/lda_lsa.RData")
load("data_derived/prod_lda.RData")

### declare training and prediction functions ----
train_classifier <- function(y, x, nodes = rep(ncol(x), 5), activation = "relu", dropout = 0.4) {
  
  x <- as.matrix(x)
  y <- data.frame(y = factor(y))
  y <- model.matrix( ~ y - 1, data = y)
  colnames(y) <- stringr::str_replace_all(colnames(y), "^y", "")
  
  
  net <- keras_model_sequential()
  
  net %>% 
    layer_dense(units = nodes[1], activation = activation, input_shape = ncol(x)) %>% 
    # layer_dropout(rate = dropout) %>% 
    layer_dense(units = nodes[2], activation = activation) %>% 
    # layer_dropout(rate = dropout) %>% 
    layer_dense(units = nodes[3], activation = activation) %>% 
    # layer_dropout(rate = dropout) %>% 
    layer_dense(units = nodes[4], activation = activation) %>% 
    # layer_dropout(rate = dropout) %>% 
    layer_dense(units = nodes[4], activation = activation) %>% 
    layer_dropout(rate = dropout) %>% 
    layer_dense(units = ncol(y), activation = "softmax")
  
  # summary(net)
  
  net %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
  
  history <- net %>% fit(
    x = x, y = y, 
    epochs = 50, 
    batch_size = 128,
    validation_split = 0.1
  )
  
  list(net = net, history = history)
}

predict_classifier <- function(object, new_data) {
  
  p <- predict(object, new_data)
  
  colnames(p) <- levels(factor(doc_class[test]))
  
  p
  
}

### fit a model with lsa embeddings ----
# m_lsa <- randomForest(y = factor(doc_class[train2]), x = lsa2)
# 
# p_lsa <- predict(m_lsa, lsa3, type = "prob")

m_lsa <- train_classifier(y = doc_class[train2],
                          x = lsa2,
                          dropout = 0.3,
                          activation = "relu")

p_lsa <- predict_classifier(object = m_lsa$net,
                            new_data = lsa3)

### fit a model with lda embeddings ----
# m_lda <- randomForest(y = factor(doc_class[train2]), x = lda2)
# 
# p_lda <- predict(m_lda, lda3, type = "prob")

m_lda <- train_classifier(y = doc_class[train2],
                          x = lda2,
                          dropout = 0.3,
                          activation = "relu")

p_lda <- predict_classifier(object = m_lda$net,
                            new_data = lda3)

### fit a model with prod_lda embeddings ----
# m_prod <- randomForest(y = factor(doc_class[train2]), x = prod_lda2)
# 
# p_prod <- predict(m_prod, prod_lda3, type = "prob") 

m_prod <- train_classifier(y = doc_class[train2],
                           x = prod_lda2,
                           dropout = 0.3,
                           activation = "relu")

p_prod <- predict_classifier(object = m_prod$net,
                             new_data = prod_lda3)


### save the results ----
save(m_prod, m_lda, m_lsa, p_prod, p_lda, p_lsa, file = "data_derived/classifiers.RData")
