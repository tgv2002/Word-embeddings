# Introduction to NLP - Assignment 2 (Word Embeddings)

### Name: Gokul Vamsi Thota
### Roll number: 2019111009

## Model checkpoints and embeddings

* Word embeddings (mathematical model) obtained using Singular Value Decomposition can be found [here](https://drive.google.com/file/d/1K-r0HZU_pCtIvXdJT9ihMXvAkfB4JD0Q/view?usp=sharing). 

* Word2vec model using CBOW with negative sampling can be found [here](https://drive.google.com/file/d/1ZCGKG2rRg1YJMf6Saut7Eh_zx-a2ThYv/view?usp=sharing).

* Word embeddings obtained using CBOW with negative sampling can be found [here](https://drive.google.com/file/d/180_2OLIqD7o9vHaBLQrnhRFlTLNbRw-A/view?usp=sharing).


## Instructions to execute

* The implementation corresponding to `SVD` is in the script `word2vec_svd.py`. It can be executed by the command: `python3 word2vec_svd.py`. The required folder path, which is a global variable, can be modified (line 20) as per requirement. 

* The implementation corresponding to `CBOW with negative sampling` is in the script `cbow_negative.py`. It can be executed by the command: `python3 cbow_negative.py`. The required, folder path, context window size, number of negative samples per window, and embedding dimensions are configurable global parameters (lines 29-33). 

* For both the above scripts, the comments in the code are to be followed regarding loading / saving the model / word embeddings (line 159 for `word2vec_svd.py` and line 230 for `cbow_negative.py`). Both the codes by default save the model and embeddings in the specified folder path. Load and save functions are utilized for self-explanatory functions.

* It is assumed that the corpus is extracted and is present in the actual json format `Electronics_5.json`, in the folder path indicated by the global variable `FOLDER_PATH`.

## Key Details

* For the SVD part, 130,000 data points (individual paragraphs) in the corpus are considered, and for the CBOW part, 75,000 data points were considered for the training process.

* Words which have occurred less than 5 times in the corpus are discarded.

* For both the models, an embedding dimension of 200 and window size of 5 (central word along with 2 words right before and after it) were used.

* CBOW model with negative sampling was trained for 13 epochs using 256 as batch size.