# Next Word Prediction with LSTM

This project trains a recurrent neural network to predict one likely next word from a sequence of words. The training corpus is Shakespeare's *Hamlet* from the NLTK Gutenberg collection. A Streamlit application loads the trained model and tokenizer so you can try predictions in a browser.

## Project at a glance

```text
Training.ipynb
    |
    |  download/read Hamlet, tokenize, make examples, train
    v
next_word_lstm.h5 + tokenizer.pickle
    |
    |  place both artifacts in models\
    v
app.py (Streamlit)
    |
    v
Enter a phrase -> predict the most likely next word
```

| Path | Purpose |
| --- | --- |
| `Training.ipynb` | Downloads the corpus, prepares sequences, defines and trains the LSTM, and demonstrates predictions. |
| `app.py` | Loads the saved model and tokenizer, accepts a phrase, and displays a one-word prediction. |
| `models/next_word_lstm.h5` | Trained Keras model used by the app. |
| `models/tokenizer.pickle` | Fitted Keras tokenizer; its word-to-index mapping must match the trained model. |
| `hamlet.txt` | Plain-text corpus produced by the notebook. |
| `nltk_data/` | Local NLTK download directory used by the notebook. |
| `requirements.txt` | Python packages needed by the project. |

## Requirements

- Windows
- Python 3.11 (the notebook metadata records Python 3.11.9)
- Internet access for the initial NLTK Gutenberg corpus download

## Setup

Run these commands from the project directory in PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, you can invoke the environment's Python directly instead:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, open `Training.ipynb` and select the project's `venv` Python interpreter as the notebook kernel.

## Run the application

With the virtual environment active, start Streamlit from the project directory:

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`). Enter a phrase in the text box and select **Predict Next Word**.

The app expects these files to exist:

```text
models\
  next_word_lstm.h5
  tokenizer.pickle
```

The model and tokenizer are a matched pair. If you train a new model, use the tokenizer created in that same training run.

## Training the model

Run the notebook cells from top to bottom. It downloads the NLTK corpus into the project's `nltk_data` folder, extracts `shakespeare-hamlet.txt`, and writes the plain text to `hamlet.txt`. Running from the project directory keeps these relative paths aligned with the project files.

The notebook's download cell specifies the local download folder:

```python
from pathlib import Path
import nltk

nltk_dir = Path(r"D:\Courses\GenAI Udemy Advanced\Next Word Prediction\nltk_data")
nltk_dir.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(nltk_dir))
nltk.download("gutenberg", download_dir=str(nltk_dir))
```

If you move the project, update that path to the new project's `nltk_data` directory (or configure a suitable path for your machine). A successful download should return `True`. Then the notebook reads and saves the corpus:

```python
from nltk.corpus import gutenberg

data = gutenberg.raw("shakespeare-hamlet.txt")
with open("hamlet.txt", "w", encoding="utf-8") as file:
    file.write(data)
```

After training, the notebook saves `next_word_lstm.h5` and `tokenizer.pickle` in its current working directory. The app instead loads them from `models\`. Copy the newly generated artifacts into that folder before running the app:

```powershell
New-Item -ItemType Directory -Force models
Copy-Item .\next_word_lstm.h5 .\models\next_word_lstm.h5 -Force
Copy-Item .\tokenizer.pickle .\models\tokenizer.pickle -Force
```

## `Training.ipynb`: data, intuition, and architecture

### 1. Corpus and vocabulary

The notebook loads the full raw text of *Hamlet* from NLTK's Gutenberg corpus and converts it to lowercase. Keras `Tokenizer` builds a vocabulary and assigns each retained word an integer ID. The notebook reserves index `0` for padding by setting the output vocabulary size to `len(tokenizer.word_index) + 1`.

This tokenizer is an essential part of the model: text must be converted to IDs using the exact same word-to-index mapping used during training.

### 2. Turning text into supervised examples

For each line, the notebook tokenizes the words and constructs every growing prefix as an input, with the next word as its label. For example, a tokenized line such as:

```text
to be or not
```

produces examples conceptually like:

```text
to                 -> be
to be              -> or
to be or           -> not
```

In the notebook's implementation, each prefix-plus-target sequence is created first. Once padded, all but the final token become `x` (the model input), and the final token becomes `y` (the target). Padding is added on the left (`padding="pre"`) so all input sequences have the same length while the most recent words remain at the end.

The vocabulary has 4,817 indexed words in the recorded model summary, plus the reserved padding index, giving the output layer 4,818 classes. The recorded input length is 13 tokens.

### 3. Train/test split

The examples are split with `train_test_split(test_size=0.2)`: 80% are used for training and 20% are passed as validation data to `model.fit`. The notebook does not set a random seed, so the exact split and results can vary between runs. It also uses that held-out split for validation during training; it is not a separate final test set.

### 4. Neural-network architecture

The notebook builds this Keras `Sequential` network:

```text
Input: 13 word IDs
  -> Embedding(vocabulary size, 100 dimensions)
  -> LSTM(150 units, return_sequences=True)
  -> Dropout(0.2)
  -> LSTM(100 units)
  -> Dense(vocabulary size, softmax)
  -> Probability distribution over the next word
```

The recorded model summary reports 1,219,418 trainable parameters.

#### What each layer contributes

- **Embedding** maps each integer word ID to a trainable 100-number vector. This gives the recurrent network a dense representation to learn from instead of treating word IDs as meaningful numeric quantities.
- **First LSTM (150 units)** processes the context in order. `return_sequences=True` preserves an output for every time step so the next recurrent layer can process the full sequence.
- **Dropout (0.2)** randomly drops a portion of activations during training, a regularization technique intended to reduce overfitting.
- **Second LSTM (100 units)** summarizes the sequence into a context representation.
- **Dense + softmax** produces one score/probability per vocabulary item. The item with the largest probability is the model's next-word prediction.

#### Intuition

At each training example, the model sees the preceding words and is asked to assign a high probability to the actual next word. The embedding and LSTM weights are adjusted to reduce categorical cross-entropy, the difference between the predicted word distribution and the correct next-word label. At inference time, the same transformation is applied to a user's phrase, and the highest-probability vocabulary item is returned.

This is a **many-to-one next-token classifier**: one context goes in, and one vocabulary-wide distribution comes out. It predicts one word per click; it does not generate a full sentence or sample alternative words.

### 5. Training and saved artifacts

The notebook trains for up to 100 epochs with the Adam optimizer and categorical cross-entropy loss. It passes the held-out split as `validation_data`, but the actual training cell does **not** configure early stopping, despite the notebook's introductory description mentioning it.

The notebook saves the trained Keras model in HDF5 format and serializes the fitted tokenizer with `pickle`. Keep these two artifacts together and do not substitute a tokenizer from a different training run.

## `app.py`: inference architecture

The app has four main steps:

1. **Load artifacts at startup.** Keras loads `models/next_word_lstm.h5`; `pickle` loads `models/tokenizer.pickle`.
2. **Collect input.** Streamlit displays a text box, defaulted to `To be or not to`, and a **Predict Next Word** button.
3. **Prepare model input.** `predict_next_word` converts the phrase to token IDs with the saved tokenizer. It keeps at most the latest 13 IDs and left-pads shorter phrases to 13 positions, matching the training input shape.
4. **Choose and display the prediction.** The model returns a softmax distribution. NumPy's `argmax` selects its highest-scoring class, and the tokenizer's word index maps that class back to a word for display.

The effective inference flow is:

```text
User phrase
  -> saved tokenizer: words to IDs
  -> trim/pad to 13 tokens
  -> saved embedding + LSTM + softmax model
  -> highest-probability vocabulary ID
  -> tokenizer: ID to word
  -> Streamlit displays the word
```

## Limitations and interpretation

- The model learns from a single play, so its vocabulary, spelling, and style are Shakespeare-specific; general modern-English performance should not be expected.
- The output is the single most likely word under the learned softmax distribution. It is not a confidence guarantee or a grammar check.
- A fixed 13-token context is used at prediction time. Older words beyond that window are discarded.
- Text preprocessing and tokenizer configuration are basic; unseen words cannot contribute meaningful context.
- The current notebook has no explicit random seed, early-stopping callback, or separate final test set, so its reported validation results should be treated as an instructional baseline.
- The notebook writes model artifacts to its working directory while the app loads them from `models\`. Copy the artifacts as described above after retraining.

## Dependencies

The project dependencies are listed in `requirements.txt`, including TensorFlow/Keras, NumPy, pandas, scikit-learn, NLTK, and Streamlit. Install them into the same Python environment selected for the notebook and used to run the app.
