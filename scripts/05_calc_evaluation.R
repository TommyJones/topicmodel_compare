################################################################################
# This script analyzes classification results 
################################################################################

rm(list = ls())

load("data_derived/classifiers.RData")
load("data_derived/20_newsgroups_formatted.RData")

### Declare an evaluation function ----

evaluate <- function(actual, predicted){
  
  # Make a confusion matrix
  confusion <- table(actual=actual, predicted=predicted)
  
  # Get the levels of our classification variable
  class_levels <- sort(union(rownames(confusion), colnames(confusion)))
  
  metrics <- lapply(class_levels, function(x){
    y <- setdiff(class_levels, x) # get the other names
    
    tp <- confusion[ x , x ]
    fp <- sum(confusion[ y , x ])
    tn <- sum(confusion[ y , y ])
    fn <- sum(confusion[ x , y ])
    
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    specificity <- tn / (tn + fp)
    fdr <- 1 - precision
    
    list(precision = precision,
         recall = recall,
         specificity = specificity,
         fdr = fdr)
    
  })
  
  names(metrics) <- class_levels
  
  list(confusion=confusion, metrics=metrics)
}



### Evaluate each model ----

e_lsa <- evaluate(actual = doc_class[test],
                  predicted = apply(p_lsa, 1, function(x) names(x)[which.max(x)][1]))

e_lda <- evaluate(actual = doc_class[test],
                  predicted = apply(p_lda, 1, function(x) names(x)[which.max(x)][1]))

e_prod <- evaluate(actual = doc_class[test],
                   predicted = apply(p_prod, 1, function(x) names(x)[which.max(x)][1]))

### Create a table of summary statistics ----
e_table <- data.frame(Precision = rep(NA, 3),
                      Recall = rep(NA, 3),
                      Accuracy = rep(NA, 3))

rownames(e_table) <- c("ProdLDA", "LDA", "LSI")

e_table$Precision <- c(mean(sapply(e_prod$metrics, function(x) x$precision)),
                       mean(sapply(e_lda$metrics, function(x) x$precision)),
                       mean(sapply(e_lsa$metrics, function(x) x$precision)))

e_table$Recall <- c(mean(sapply(e_prod$metrics, function(x) x$recall)),
                    mean(sapply(e_lda$metrics, function(x) x$recall)),
                    mean(sapply(e_lsa$metrics, function(x) x$recall)))

e_table$Accuracy <- c(sum(diag(e_prod$confusion)) / sum(e_prod$confusion),
                      sum(diag(e_lda$confusion)) / sum(e_lda$confusion),
                      sum(diag(e_lsa$confusion)) / sum(e_lsa$confusion))

### Create tables breaking down performance by class ----
e_class_prod <- as.data.frame(do.call(rbind, lapply(e_prod$metrics, unlist)))

e_class_lda <- as.data.frame(do.call(rbind, lapply(e_lda$metrics, unlist)))

e_class_lsa <- as.data.frame(do.call(rbind, lapply(e_lsa$metrics, unlist)))


### save results ----

save(e_table, e_class_prod, e_class_lda, e_class_lsa, e_lda, e_lsa, e_prod,
     file = "data_derived/evaluation.RData")

