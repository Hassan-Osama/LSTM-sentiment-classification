# Sentiment Classification Using LSTM model and GLOVE pretrained embedding model

---

## Work Flow

### **1.** Data PreProcessing (data_preprocessing.py)
* Load raw data
* Clean reviews column
    * Remove links
    * Convert to lower case
    * Remove non English characters
    * Lematize (convert adjs and verbs to their basic form)
    * Add cleaned reviews to a new column **clean_review**
* Save the cleaned data in data/processed

### **2.** Build Vocab (build_vocab.py)
* Load processed dataset
* Build vocab Map (to be used in the embeddings)
    * Special Vocabs
        * <PAD> used to pad the sequence to make all sequences with a constant dimension
        * <UNK> used when we get an unknown word from our vocab
* Save the vocab list in data/processed as json

### **3.** Embedding (embeddings.py)
* Uses a pretrained embedding model **GLOVE** to create embedding matrix from our vocab list
* Save the embedding matrix in data/processed

### **4.** Creating Data loaders (dataset.py)
* Crated data loader to be used in model training

### **5.** Model (model.py)
* LSTMSentimentClassifier that has:
    * Embedding Layer(embed using the embeding matrix we created using the pretrained GLOVE model)
    * LSTM Layer
    * DropOUT Layer(to not overfit)
    * Fully Connected Layer
    * Segmoid Activation Function(activation function that convert the percentage comming out from the FC layer from a percentage to two values: 1 or 0 representing each class)

### **6.** Model Training and Evaluation (train.py)
* Train the model on 5 epochs
* validate each epoch
* Save the model in modles every epoch if the F1 score of the validation(to prevent saving an overfitted model) is higher than the highest model we encountered

---

## Data Exploration

### The Data has two columns "review" and "sentiment"

### Sentiment Distribution
![alt text](https://github.com/Hassan-Osama/LSTM-sentiment-classification/blob/main/figures/sentimen_distribution.png?raw=true)

### Review length distribution
![alt text](https://github.com/Hassan-Osama/LSTM-sentiment-classification/blob/main/figures/review_length_distribution.png?raw=true)

### Word Cloud
![alt text](https://github.com/Hassan-Osama/LSTM-sentiment-classification/blob/main/figures/positive_reviews_word_cloud.png?raw=true)
![alt text](https://github.com/Hassan-Osama/LSTM-sentiment-classification/blob/main/figures/negative_reviews_word_cloud.png?raw=true)

---

## Training and Validation Scores
| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1  | Best Model Saved |
|-------|------------|-----------|----------|----------|---------|---------|------------------|
| 1     | 0.6923     | 0.5077    | 0.4896   | 0.6917   | 0.5062  | 0.2069  | Yes              |
| 2     | 0.5887     | 0.6518    | 0.6049   | 0.3400   | 0.8610  | 0.8574  | Yes              |
| 3     | 0.2692     | 0.8936    | 0.8938   | 0.2679   | 0.8930  | 0.8948  | Yes              |
| 4     | 0.1596     | 0.9421    | 0.9420   | 0.2892   | 0.8874  | 0.8860  | No               |
| 5     | 0.0896     | 0.9717    | 0.9717   | 0.3228   | 0.8866  | 0.8888  | No               |

---

