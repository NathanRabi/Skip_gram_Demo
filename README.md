# Word2Vec Skip-Gram Demo

An interactive GUI application that trains and explores a Skip-Gram Word2Vec model on a subset of the **text8** corpus.

**Course:** Data Science at Tel Aviv University   
**Authors:** Nathan Rabinovich, Yuval Levi

---

##  Features

- **Skip-Gram** with Negative Sampling (efficient training)
- **Nearest-neighbour** lookup for any word or vector arithmetic expression
- **2D PCA** projection of embeddings for visualization
- **Progress bar** via Gensim callback during training
- **Intrinsic Evaluation** via Google's analogy accuracy test
- Lightweight **Tkinter** GUI for interactive exploration

---

##  Prerequisites

- Python 3.7+  
https://www.python.org/downloads/release/python-3133/

- Install dependencies:
pip install gensim scikit-learn matplotlib numpy tqdm

##  Installation


Clone this repository:
 -Press Win+r, type CMD and press Enter.
  git clone https://github.com/NathanRabi/word2vec-skipgram-demo.git
 - or install the file from the github
- cd word2vec-skipgram-demo (python script path)
Install Python packages: 
- pip install gensim scikit-learn matplotlib numpy tqdm

##  Usage
Run the application:

- python word2vec_demo.py

Wait for the background training to finish.
Once the entry box is enabled, type a word (or a vector expression like king - man + woman) and press Enter.

View the nearest neighbours in the text panel and see their 2D PCA plot on the right.

---

##  Configuration
At the top of word2vec_demo.py you can adjust:

python

SENTENCES = 10000    # Number of sentences from text8

DIM       = 100     # Embedding dimensionality

EPOCHS    = 3       # Training epochs

MIN_COUNT = 5       # Minimum word frequency

WINDOW    = 5       # Context window size

TOP_N     = 20      # Neighbours to display

---

## File Structure

├── word2vec_demo.py      # Main application

├── requirements.txt      # pinned dependencies

└── README.md             # This file

---

## Screenshots

Training Output

![image](https://github.com/user-attachments/assets/fd82d8bf-5db3-4582-baa6-67c1b56ecc5c)

Model Evaluation 

![image](https://github.com/user-attachments/assets/03cded5c-b774-41d7-baf4-f5c2332a3456)

Active GUI (After Training Completes)

![image](https://github.com/user-attachments/assets/b2afca04-bd45-437b-be61-0088556a87de)

Nearest Neighbours Output

![image](https://github.com/user-attachments/assets/c454566e-ff93-47af-b8e8-db4c46b9b135)

2D PCA Plot Visualization

![image](https://github.com/user-attachments/assets/e3e3c257-47d2-4788-a798-1bc59c7d3da3)



## References

### Word2Vec  
1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). *arXiv preprint arXiv:1301.3781*.  
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546). *arXiv preprint arXiv:1310.4546*.

### Word2Vec Explanation  
3. Goldberg, Y., & Levy, O. (2014). [word2vec Explained: deriving Mikolov et al.’s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722). *arXiv preprint arXiv:1402.3722*.

### Linguistic Theory  
4. Firth, J. R. (1957). *A Synopsis of Linguistic Theory, 1930–1955*. In J. R. Firth (Ed.), *Studies in Linguistic Analysis* (pp. 1–32). Blackwell, Oxford.
