# Word2Vec Skip-Gram Demo

An interactive GUI application that trains and explores a Skip-Gram Word2Vec model on a subset of the **text8** corpus.

**Course:** Data Science at Tel Aviv University   
**Authors:** Nathan Rabinovich, Yuval Levi

---

## 🚀 Features

- **Skip-Gram** with Negative Sampling (efficient training)
- **Nearest-neighbour** lookup for any word or vector arithmetic expression
- **2D PCA** projection of embeddings for visualization
- **Progress bar** via Gensim callback during training
- **Intrinsic Evaluation** via Google's analogy accuracy test
- Lightweight **Tkinter** GUI for interactive exploration

---

## 📦 Prerequisites

- Python 3.7+  
- Install dependencies:

pip install gensim scikit-learn matplotlib numpy tqdm
🔧 Installation
Clone this repository

git clone https://github.com/<your-username>/word2vec-skipgram-demo.git
cd word2vec-skipgram-demo
Install Python packages (see above).

▶️ Usage
Run the application:


python3 word2vec_demo.py
Wait for the background training to finish.

Once the entry box is enabled, type a word (or a vector expression like king - man + woman) and press Enter.

View the nearest neighbours in the text panel and see their 2D PCA plot on the right.

⚙️ Configuration
At the top of word2vec_demo.py you can adjust:

python

SENTENCES = 10000    # Number of sentences from text8

DIM       = 100     # Embedding dimensionality

EPOCHS    = 3       # Training epochs

MIN_COUNT = 5       # Minimum word frequency

WINDOW    = 5       # Context window size

TOP_N     = 20      # Neighbours to display

📁 File Structure

├── word2vec_demo.py      # Main application

├── requirements.txt      # pinned dependencies

└── README.md             # This file


📚 References
Word2Vec
[1] Mikolov, T., Chen, K., Corrado, G., Dean, J. 2013, 'Efficient Estimation of Word Representations in Vector Space', arXiv preprint arXiv:1301.3781.
Word2vec Explanation
[2] Goldberg, Y., Levy, O. 2014, 'word2vec Explained: deriving Mikolov et al.'s negative-sampling
  word-embedding method', arXiv preprint arXiv:1402.3722.
Linguistic Theory
[3] Firth, J.R. (1957) A Synopsis of Linguistic Theory, 1930-1955. In: Firth, J.R., Ed., Studies in Linguistic Analysis, Blackwell, Oxford, 1-32
