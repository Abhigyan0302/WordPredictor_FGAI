# WordPredictor_FGAI

An autoregressive word-level language modeling project that implements and compares multiple Recurrent Neural Network architectures (RNN, BiRNN, LSTM, and GRU) using PyTorch. The primary objective is to predict the next word in a sequence and evaluate the comparative performance of these models on a synthetic story dataset.

## Dataset
The model is trained on `tinystories_small.txt`, a subset of the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. 

**Preprocessing pipeline:**
* All text is converted to lowercase and split into word-level tokens.
* A vocabulary dictionary is built mapping unique tokens to indices, including a `<pad>` token for sequence alignment.
* Training sequences are generated using a sliding window approach with a window size of 30, resulting in zero-padded sequences up to a maximum length of 31.
* The final processed dataset yields **1,690,707 sequences**, divided into an 80/20 train-validation split.

## Architectures Implemented
The project utilizes a custom `AutoregressiveModel` module featuring an embedding layer (dimension = 128) and a hidden layer (dimension = 256), mapped to an output fully-connected layer matching the vocabulary size. 

Four recurrent variations are evaluated:
1. **Standard RNN:** Forward-processing Recurrent Neural Network.
2. **BiRNN:** Bidirectional RNN (implemented as a comparative baseline, though inherently misaligned with causal next-word prediction).
3. **LSTM:** Long Short-Term Memory network utilizing memory cells and gates for long-term dependencies.
4. **GRU:** Gated Recurrent Unit offering a simplified gating mechanism compared to LSTM.

## Setup and Execution

### Prerequisites
* Python 3.x
* PyTorch
* NumPy, Pandas, Matplotlib
* Scikit-learn

### Running the Project (Google Colab)
1. Clone the repository and upload `tinystories_small.txt` to your Colab environment.
2. Execute the preprocessing cells to tokenize the text and generate the sliding window sequences.
3. Load the PyTorch `DataLoader` (Batch Size = 128).
4. Train the models. The training loop utilizes the **Adam optimizer (lr = 0.001)** and **CrossEntropyLoss**. 
5. Models are trained for up to 50 epochs with an **Early Stopping patience of 7 epochs**. 
6. (Optional) Requires a T4 GPU hardware accelerator for optimal training time.

## Results and Evaluation
The advanced gated architectures (LSTM and GRU) significantly outperformed the simpler recurrent models. Below is the final evaluation captured before early stopping:

| Model | Train Time (s) | Validation Loss | Validation Accuracy | Perplexity |
| :--- | :--- | :--- | :--- | :--- |
| **RNN** | 1004.47 | 4.1632 | 0.2887 | 64.2748 |
| **BiRNN** | 1455.84 | 4.2680 | 0.2793 | 71.3792 |
| **LSTM** | 1242.55 | **3.7142** | **0.3405** | **41.0244** |
| **GRU** | 1194.54 | 3.7158 | 0.3365 | 41.0929 |

**Conclusion:** The **LSTM** proved to be the best-performing model, achieving the highest accuracy and lowest perplexity. The **GRU** performed as a highly effective runner-up, delivering near-identical predictive accuracy while completing its training marginally faster. As expected, the BiRNN performed the worst, reinforcing that bidirectional information flow is not suited for causal text generation tasks.

---
**Author:** Abhigyan Singh  
*B.Tech in Computer and Communication Engineering, Manipal Institute of Technology*
