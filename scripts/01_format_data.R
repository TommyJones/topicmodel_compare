################################################################################
# This script splits data into two training sets and a test set
################################################################################

rm(list = ls())

### Load necessary libraries ----
library(textmineR)

library(stringr)

### load 20 newsgroups data and build dtm ----
if (! file.exists("news20")) { # if statement so I don't duplicate download etc.
  download.file("http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
                destfile = "news20.tar.gz")
  
  untar(tarfile = "news20.tar.gz",
        exdir = "news20/")
}



docnames <- list.files("news20/20_newsgroup", full.names = TRUE, recursive = TRUE)

docs <- parallel::mclapply(docnames, function(d){
  doc <- scan(d, what = "character", sep = "\n")
  
  doc <- paste(doc, collapse = "\n")
  
  doc
}, mc.cores = 4) %>% 
  unlist() %>%
  stringr::str_conv("UTF-8")

names(docs) <- docnames

doc_class <- stringr::str_split(docnames, pattern = "/") %>%
  sapply(function(x) x[3])

dtm <- CreateDtm(doc_vec = docs) # stopwords English and SMART

dim(dtm)

# prune vocab so that only words appearing in greater than 5 docs are included
dtm <- dtm[, colSums(dtm > 0) >= 5]

dim(dtm)

### sample rows into three groups ----

idx <- seq_len(nrow(dtm))

train1 <- sample(idx, 6665)

train2 <- sample(setdiff(idx, train1), 6665)

test <- setdiff(idx, c(train1, train2))

### save the things we need for future use ----
save(docs, dtm, train1, train2, test, doc_class, file = "data_derived/20_newsgroups_formatted.RData")

### remove unzipped newsgroups files ----
# (keeps them from syncing with my cloud backup)

system("rm -r news20")
