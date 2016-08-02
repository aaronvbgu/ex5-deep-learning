EX5: Deep Learning
==================



RNN Based Text Generator
------------------------

Our goal is to build and train a model which will be able to predict the next word.



**Steps**
  1. Text tokenization using pythons nltk library
  2. Add our special tokens
  3. Mapping words to vectors (words to indexes)
  4. Initialize RNN using pythons numpy library
  5. Perdict words probabilities
  


**How can we improve the process?**
  1. Remove unnecessary words (replacing uncommon or rare words with special tokens)



**Communities(Short Random Walks)**
  1. Modularity: 0.765
  2. Sizes: 13 7 5 5 7 5 5
  3.![grpah1](imdb-graph1.png)


**Communities(Greedy Optimization)**
  1. Modularity: 0.765
  2. Sizes: 7 5 13 5 5 5 7
  3.![grpah1](imdb-graph2.png)