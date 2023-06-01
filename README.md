# Hate Speech Detection in Hinglish and English using Deep Learning
This repository contains a Deep Learning model that can detect and classify hate speech in Hinglish and English text. The model was trained on a dataset of approximately 15,000 Hinglish and English tweets, and has an accuracy of around 86% on a test set. 

# Requirements
- PyTorch
- Pandas
- Datasets
- Transformers
- wandb

You can install these using pip:
```
pip install pandas datasets transformers wandb

```

You can install pytorch from [here](https://pytorch.org/get-started/locally/).

You can create your wandb account from [here](https://wandb.ai/site)

This model was built on top of [MuRIL](https://huggingface.co/google/muril-large-cased) and requires a GPU to train. 

# Usage
To use the model, simply clone this repository and open Eng+Hing_model.ipynb on your text editor. Run all the cells till you reach **LOAD SAVED MODEL**. After completing the training process, load your model and input your preferred text. 

The output will be either one of the two:
- 'Hate Speech!'
- 'Not Hate Speech!'

# About the Model
One of the key strengths of the MuRIL BERT model is its ability to understand the Indian political context and catch subtleties in hate speech that may be unique to India. By fine-tuning the MuRIL BERT model on a dataset of labeled hate speech examples, the hate speech detection and classification model is able to effectively detect and classify hate speech in Hinglish and English text, including hate speech that may be disguised or nuanced. This is particularly important given the complex sociopolitical landscape of India and the prevalence of hate speech in political discourse.

For example, the model can pick up on subtle cues in hate speech related to caste, religion, and regional identity, which are often intertwined with politics in India. This is critical for accurately detecting and classifying hate speech in the Indian context, and highlights the model's ability to capture the nuances and complexities of language use in India.


# Model Details
This model was built on top of the Multilingual Representations for Indian Languages (MuRIL) BERT model, which is a pre-trained language model developed specifically for Indian languages, and covers around 17 Indic languages including Hinglish (Hindi + English code) and English. To build the hate speech detection and classification model, a classifier layer was added on top of the MuRIL BERT model. To fine-tune the model for the hate speech detection task, it was trained on a dataset of labeled examples covering a variety of controversial topics such as Castiesm, misogyny, Islamophobia, Hinduphobia, Racism and much much more.

During training, the model was optimized using the binary cross-entropy loss function and the Adam optimizer. To further improve the accuracy and performance of the model, the three BERT models in the MuRIL BERT model - the base, large, and small models - were frozen during training. This helped to reduce the number of trainable parameters in the model and prevented overfitting, leading to better generalization performance.

This model has achieved an accuracy of around 86% on a test set and can be used to detect and classify hate speech in both Hinglish and English text.

# Note
You might have to manually create the **models** folder in your directory if it is not already present.
