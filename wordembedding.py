#!/usr/bin/env python3
# Word2Vec Skip-Gram  Demo - Data Science seminar - Nathan Rabinovich, Yuval Levi.
# This script trains a small Word2Vec model on a subset of the text8

"""

Word2Vec Skip-Gram Demo 
======================

Authorship
------------
Nathan Rabinovich, Yuval Levi

AI Assistance 
------------
 OpenAI ChatGPT o3 - refactoring, Logging
 Google Gemini 2.5 Pro Preview (temp 0.5) - UI, Debugging  

Data
------------
 **text8** - cleaned English Wikipedia dump  
  © Matt Mahoney 2006  ·  Public Domain  
  loaded via `gensim.downloader`

Libraries
-------------
 gensim 4.x, scikit-learn 1.x, matplotlib 3.x, numpy 1.x, tqdm 4.x  

 *pip install gensim tqdm scikit-learn matplotlib numpy

Refrences:
--------------
Dieng A.B., Ruiz F.J.R., Blei D.M.  
*Topic Modeling in Embedding Spaces.* arXiv:1907.04907 (2019)

"""

import os
import sys
import threading
import itertools
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import gensim.downloader as api
from gensim.models import Word2Vec, callbacks
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import patheffects
from tqdm import tqdm # Progress bar
import re

# ---------Configuration  ---------
# Word2Vec Training Parameters
SENTENCES = 5000    # Number of sentences from word list to use for training
DIM       = 100      # Dimensionality of the word vectors
EPOCHS    = 3        # Number of training epochs (passes over the data)
MIN_COUNT = 5        # Minimum word frequency; words below this are ignored
WINDOW    = 5        # Max distance between current & predicted word in sentence

# Exploration Parameters
TOP_N = 20        # Number of nearest neighbours to find and plot

# Plotting Style
PLOT_CMAP           = 'coolwarm_r'    # Colormap  
TARGET_MARKER_COLOR = 'red'           # maker color
TARGET_MARKER_SHAPE = 'o'             # Marker Shape
TARGET_MARKER_SIZE  = 130             # Marker Size

# Logging Setup
logging.basicConfig(format="%(levelname)-8s %(asctime)s %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# --- Helper Class for Training Progress ---
class EpochProgress(callbacks.CallbackAny2Vec):
    """ Callback to display progress bar during Word2Vec training. """
    def __init__(self, total_epochs: int):
        self.pbar = tqdm(total=total_epochs, desc="Training",
                         ncols=70, unit="ep", file=sys.stdout)
    def on_epoch_end(self, model):
        self.pbar.update(1)
    def on_train_end(self, model):
        self.pbar.close()

# --- Helper Function for Parse ---
def parse_vector(expr, wv):
    """ 
    expr: " a - b = c"
    wv: your KeyedVectors object
    returns: a target vector or None if paring fails 
    """
    
    tokens = expr.replace(" ", "").split("+")
    vec = None
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-")
            vec = (wv[a] - wv[b]) if vec is None else vec + (wv[a] - wv[b])
        else:
            vec = wv[tok] if vec is None else vec + wv[tok]
    return vec

# --- Main Application Class ---
class Word2VecApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Word2Vec Demo")
        root.geometry("830x580")
        self.wv: Optional[KeyedVectors] = None # hold the trained Word2Vec KeyedVectors

        # --- UI Setup ---
        # Controls (Word Entry)
        ctrl = ttk.Frame(root, padding=(10, 8)); ctrl.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl, text="Word:").pack(side=tk.LEFT)
        self.entry = ttk.Entry(ctrl, width=25, state="disabled",
                               font=("Segoe UI", 10))
        self.entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.entry.bind("<Return>", self._trigger_analysis) # Trigger on Enter key

        # Main (Split Panes)
        panes = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        panes.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                   padx=10, pady=(0, 10))

        # Neighbour List
        left = ttk.Frame(panes, padding=5); panes.add(left, weight=1)
        left.rowconfigure(1, weight=1); left.columnconfigure(0, weight=1)
        ttk.Label(left, text="Nearest Neighbours").grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 3))
        self.txt = tk.Text(left, width=28, height=15, font=("Consolas", 10),
                           state="disabled", wrap="none", borderwidth=0,
                           relief=tk.FLAT)
        scr_y = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.txt.yview)
        scr_x = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.txt.xview)
        self.txt.config(yscrollcommand=scr_y.set, xscrollcommand=scr_x.set)
        self.txt.grid(row=1, column=0, sticky="nsew")
        scr_y.grid(row=1, column=1, sticky="ns")
        scr_x.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Embedding Plot
        right = ttk.Frame(panes, padding=5); panes.add(right, weight=3)
        self.fig: Figure = plt.Figure(figsize=(5, 4.5), dpi=100)
        self.ax: Axes   = self.fig.add_subplot(111)
        self.ax.set_xlabel("PCA 1", fontsize=9) # X axis
        self.ax.set_ylabel("PCA 2", fontsize=9) # Y axis
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw() # Initial empty plot draw

        # --- Start Background Training ---
        log.info("Launching background training thread...")
        threading.Thread(target=self._train_model, daemon=True).start()

    # --- Background Training Method ---
    def _train_model(self):
        """ Loads data, trains the Word2Vec model & evaluates it using Google's analogy accuracy test"""
        try:
            log.info(f"Loading {SENTENCES:,} sentences from text8 dataset...")
            # text8 benchmark dataset (first 100MB of cleaned Wikipedia)
            sentences = list(itertools.islice(api.load("text8"), SENTENCES))
            log.info(f"Training Word2Vec model ({DIM} dims, {EPOCHS} epochs)...")

            # Instantiate and train the Word2Vec model
            model = Word2Vec(
                sentences,
                vector_size=DIM,      # Dimensionality of the word vectors.
                window=WINDOW,        # Max distance current and predisted word within sentence.
                min_count=MIN_COUNT,  # Ignores words with frequency lower than this.
                sg=1,                 # Use Skip-Gram (predicts context from word).
                negative=10,          # Negative Samples - How many "noise words" should be drawn.
                workers=max(1, os.cpu_count()-1), # Use available CPU cores for parallel training.
                epochs=EPOCHS,        # Number of iterations (epochs) over the corpus.
                seed=1,              # Random seed for reproducibility.
                callbacks=[EpochProgress(EPOCHS)] # Progress bar callback.
            )

            # Store the trained KeyedVectors (vectors and vocab)
            self.wv = model.wv
            log.info(f"Vocabulary ready: {len(self.wv):,} unique tokens.")

            # --- Intrinsic evaluation ---
            from gensim.test.utils import datapath

            # Google analogy accuracy (king – man + woman = queen)
            analogies, sections = self.wv.evaluate_word_analogies(
                datapath("questions-words.txt")
            )
            log.info(f"Google analogy accuracy: {analogies*100:.1f}%")

            # Update UI from the main thread 
            self.root.after(0, lambda: self.entry.config(state="normal"))
            self.root.after(0, self.entry.focus)

        except Exception as e:
            log.exception("Word2Vec training failed")
            self.root.after(0, lambda: messagebox.showerror(
                "Training Error", f"Could not train model.\n{e}"))

    # --- UI Event Handlers ---
    def _trigger_analysis(self, _=None):
        """ Handles word/parse entry """
        if not self.wv:
            messagebox.showwarning("Model Not Ready")
            return

        word = self.entry.get().strip().lower() # Get and clean input word
        
        # Check for parse "a - b = c"
        
        if "+" in word or "-" in word:
            try:
                # 1. strip out any quotes or punctuation except +/-
                expr = re.sub(r'[\"\'\.,]', '', word)
                # 2. find sign+token pairs (sign is "" for plain words)
                parts = re.findall(r'([+-]?)([A-Za-z0-9_]+)', expr)
                positive, negative = [], []
                for sign, tok in parts:
                    if sign == "-":
                        negative.append(tok)
                    else:
                        positive.append(tok)
                # 3. call gensim’s most_similar
                sims = self.wv.most_similar(positive=positive,
                                            negative=negative,
                                            topn=TOP_N)
                self._analyse_parsed(expr, sims)
            except KeyError as oov:
                messagebox.showerror("OOV", f"'{oov.args[0]}' is not in the vocabulary.")
                return
            return
        
        # Clear previous results
        self.txt.config(state="normal"); self.txt.delete("1.0", tk.END)
        self.txt.config(state="disabled")
        self.ax.cla() # Clear the plot axes
        self.ax.set_xlabel("PCA Dimension 1", fontsize=9) # Reset labels/title
        self.ax.set_ylabel("PCA Dimension 2", fontsize=9)
        self.ax.grid(False)
        self.ax.set_title("")
        self.canvas.draw()

        if not word: # If entry is empty, do nothing more
            return

        # Check if the word exists in the model's vocabulary
        if word not in self.wv:
            messagebox.showerror("Word Not Found",
                                 f"The word '{word}' is not in the model's vocabulary.")
            return

        # If word is valid, proceed with analysis
        self._analyse_word(word)
    
    def _analyse_parsed(self, label: str, sims):
        """ Parsed Phrases helper function"""
        # 1 · update the text box
        self.txt.config(state="normal")
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, f"Neighbours of '{label}':\n" + "-"*20 + "\n")
        for w, sc in sims:
            self.txt.insert(tk.END, f"{w:<15} {sc:.4f}\n")
        self.txt.config(state="disabled")

        # 2 · plot
        if sims:
            self._analyse_word(sims[0][0])

    def _analyse_word(self, word: str):
        """ Finds neighbours, performs PCA, and updates the UI. """
        try:
            # 1. Find Most Similar Words
            # `most_similar` returns list of (word, cosine_similarity) tuples
            similar_words = self.wv.most_similar(word, topn=TOP_N)
            if not similar_words:
                messagebox.showinfo("No Neighbours",
                                     f"No similar words found for '{word}' "
                                     f"(thresholds might be too high or word is isolated).")
                return

            # 2. Update Text Box with Neighbours
            self.txt.config(state="normal") # Enable writing
            self.txt.insert(tk.END, f"Neighbours of '{word}':\n{'-'*20}\n")
            for w, score in similar_words:
                self.txt.insert(tk.END, f"{w:<15} {score:.4f}\n") # Padded output
            self.txt.config(state="disabled") # Disable writing

            # 3. Prepare Data for PCA Plotting
            # Include the target word itself in the list for PCA
            labels = [word] + [w for w, _ in similar_words]
            # Get the high-dimensional vectors for these words
            vectors = np.array([self.wv[w] for w in labels])

            # 4. Perform PCA
            # Reduce dimensionality from DIM to 2 for visualization
            pca = PCA(n_components=2, random_state=42) # Use fixed seed for reproducibility
            coords = pca.fit_transform(vectors)
            target_coords, neighbour_coords = coords[0], coords[1:] # Separate target word

            # 5. Plot PCA Results
            self.ax.cla() # Clear previous plot elements

            # Plot neighbours, coloring by distance from target in PCA space
            if neighbour_coords.size > 0:
                # Calculate Euclidean distance in 2D PCA space
                distances = np.linalg.norm(neighbour_coords - target_coords, axis=1)
                # Normalize distances [0, 1] for colormap (handle case of single neighbour/zero range)
                min_d, max_d = distances.min(), distances.max()
                norm_distances = np.zeros_like(distances) if abs(max_d - min_d) < 1e-6 \
                                 else (distances - min_d) / (max_d - min_d)
                # Scatter plot for neighbours
                scatter = self.ax.scatter(neighbour_coords[:, 0], neighbour_coords[:, 1],
                                          c=norm_distances, cmap=PLOT_CMAP,
                                          s=50, alpha=0.85, vmin=0.0, vmax=1.0, zorder=5)
                # Optional: Add a colorbar (can clutter small plots)
                # self.fig.colorbar(scatter, ax=self.ax, label='Normalized PCA Distance')

            # Plot the target word marker on top
            self.ax.scatter(target_coords[0], target_coords[1], c=TARGET_MARKER_COLOR,
                            s=TARGET_MARKER_SIZE, marker=TARGET_MARKER_SHAPE,
                            edgecolors='black', linewidth=0.6, zorder=10) # zorder puts it on top

            # Add text labels to points
            # Calculate small offsets based on plot range to avoid labels overlapping points
            x_range = np.ptp(coords[:, 0]) or 1.0 # Avoid division by zero if all points coincide
            y_range = np.ptp(coords[:, 1]) or 1.0
            x_offset, y_offset = x_range * 0.01, y_range * 0.01
            for i, label in enumerate(labels):
                self.ax.text(
                    coords[i, 0] + x_offset,
                    coords[i, 1] + y_offset,
                    f" {label}",
                    fontsize=9,
                    color='black',
                    zorder=15
            )
            # Finalize plot appearance
            self.ax.set_title(f"PCA Projection of '{word}' and Neighbours", fontsize=10)
            self.ax.set_xlabel("PCA Dimension 1", fontsize=9)
            self.ax.set_ylabel("PCA Dimension 2", fontsize=9)
            self.ax.grid(True, linestyle='--', alpha=0.4, zorder=0) # Grid Style
            self.ax.tick_params(axis='both', which='major', labelsize=8)
            self.fig.tight_layout(pad=1.5) 
            self.canvas.draw() # Redraw the canvas with the new plot

        except Exception as e:
            log.exception(f"Analysis error for word '{word}'")
            messagebox.showerror("Analysis Error",
                                 f"Could not analyse '{word}'.\n{e}")

#  --------- Application Main  ---------
if __name__ == "__main__":
    log.info("Starting Word2Vec Demo")
    tk_root = tk.Tk()       # Create main Tkinter window
    app = Word2VecApp(tk_root) # Instantiate application class
    tk_root.mainloop()      # Start Tkinter event loop
    log.info("Application closed.") # If you are reading this, I hope you are doing well :)
