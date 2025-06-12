# Deep Learning Applications Laboratory #3 - README

## Overview

Welcome! This README walks through Laboratory #3 for Deep Learning Applications, where I explored the transformer architecture and the Hugging Face ecosystem. 

## Exercise 1.1:

I loaded and explored the Cornell Rotten Tomatoes movie review dataset, focusing on understanding its splits (train, validation, test). Using HuggingFace's datasets library, I confirmed the balanced nature of the dataset, which contains equal positive and negative reviews.

---

## Exercise 1.2: 

I loaded the lightweight Distilbert model and corresponding tokenizer, examining their outputs on sample texts. The tokenizer provided inputs in the required format (input IDs and attention masks), and Distilbert's output included hidden states, notably the [CLS] token at the start of each sequence.

---

## Exercise 1.3: 
I extracted features from Distilbert's [CLS] token, effectively turning the model into a feature extractor. These features were then used to train an SVM classifier. Using GridSearchCV, I found optimal hyperparameters (C, kernel, gamma) that yielded a solid baseline accuracy on validation data. 

---

## Exercise 2.1:
The dataset was tokenized efficiently using the map method from HuggingFace's Dataset class. Tokenization provided input IDs and attention masks, preparing the dataset effectively for training.

---

## Exercise 2.2:
I initialized Distilbert specifically for sequence classification by attaching a randomly-initialized classification head, preparing the model directly for sentiment analysis tasks.

---

## Exercise 2.3: 
Using HuggingFace's Trainer, I fine-tuned the Distilbert model. A DataCollatorWithPadding ensured efficient batch handling. The training setup included early stopping and careful hyperparameter selection (epochs, batch size, learning rate) managed via TrainingArguments. Metrics computed included accuracy, precision, recall, and F1 score, logged through WandB. 

---

## Exercise 3.3:
### Medical Image Dataset Analysis: Squeezing Every Drop Out of CLIP Features

#### Dataset and Setup

I worked with a publicly available chest X-ray dataset (NIH Chest X-ray Dataset), which came with images and labels for multiple pathologies. After checking which images I actually had downloaded (I was **NOT** downloading 45 GB of chest X-rays), I filtered the dataset accordingly.

#### The Main Idea: Milk CLIP for All It's Worth

CLIP, specifically Stanford’s XraySigLIP model, was my feature extractor of choice.  
I explored three flavors of models on top of CLIP's embeddings:

##### 1. Baseline MLP

The baseline was simple but necessary: it took the image features extracted by CLIP’s vision encoder and fed them into a Multi-Layer Perceptron (MLP) for multilabel classification.

**Architecture details:**

- Input dimension: equal to CLIP’s image feature vector size.
- Hidden layers: a sequence of fully connected layers with decreasing size.
- Regularization: dropout layers and batch normalization to prevent overfitting.
- Output layer: one neuron per pathology with sigmoid activation (implemented as BCEWithLogitsLoss during training).

**Training strategy:**

- Extracted features on the fly from the dataset images.
- Trained with binary cross-entropy for multilabel tasks.
- Added a tiny Gaussian noise to inputs during training.
- Used learning rate scheduling.

**Why baseline?**

This model acted as a sanity check: can CLIP’s raw image features alone classify these conditions well? Also, it set a performance floor to improve upon.

---

##### 2. Two-Branch MLP with Concatenation Fusion

Here’s where things got interesting. Instead of relying solely on image features, I also exploited CLIP’s powerful text encoder. The idea: each pathology had a set of descriptive prompts (like “signs of pneumothorax”) which got encoded into vectors.

**How it worked:**

- For each pathology, generated multiple positive prompts using templates such as:
  - “signs of {pathology}”
  - “evidence of {pathology}”
  - “presence of {pathology}”
- Encoded these prompts through CLIP’s text encoder and averaged their embeddings to get a robust text representation per pathology.
- Stacked these per-pathology prompt embeddings into a “mega-vector” representing all pathologies’ textual context.
- At inference/training, the model received two inputs:
  1. The image feature vector from CLIP.
  2. The flattened prompt mega-vector.
- The two inputs went through separate branches (fully connected layers with ReLU, batch norm, dropout).
- Their resulting embeddings got concatenated and passed through a final classifier layer.

**Why concatenation fusion?**

Concatenating image and text embeddings allowed the model to learn joint patterns. For example, it could learn that certain textual cues emphasize some image features, helping it disambiguate challenging cases.

**Technical notes:**

- The prompt mega-vector was fixed during training (not learned) and acted as a strong prior.
- The model had separate parameter sets for image and prompt branches, enabling specialized transformations.
- The final classifier was a simple MLP that took the concatenated features and output pathology probabilities.
---

##### 3. Weighted Sum MLP

This approach tried to be smarter about combining image and text features by introducing learnable weights that decide how much emphasis to put on each modality’s features.

**Mechanics:**

- Took the prompt mega-vector (same as in the two-branch model).
- Projected it via a linear layer (with ReLU) to match the image feature dimension.
- Introduced two learnable vectors of weights: one for image features, one for prompt features.
- Computed a weighted sum:
  
  \[
  \text{combined\_features} = w_{img} \odot \text{image\_features} + w_{prompt} \odot \text{projected\_prompt\_features}
  \]

  where \(\odot\) is element-wise multiplication.

- Passed combined features through an MLP similar in size to the baseline’s for classification.

**Why weighted sum?**

- Instead of blindly concatenating, the model dynamically learned which dimensions of image and text features deserved more attention.
- This helped reduce redundancy and noise, as the model could “mute” less relevant features.

**Additional remarks:**

- The weights \(w_{img}\) and \(w_{prompt}\) were learned end-to-end during training.
- This setup still used the same prompt mega-vector ensemble as before.
- The prompt projection aligned textual features into the same space as image features for meaningful element-wise combination.

#### How I Made the Prompts (Prompt Engineering)

For each pathology, I crafted multiple positive prompts using templates like:

- “signs of {pathology}”
- “indication of {pathology}”
- “evidence of {pathology}”
- “presence of {pathology}”

These got fed to CLIP’s text encoder and averaged to form a single prompt embedding per pathology.

#### Training and Evaluation Highlights

- Trained each model on balanced subsets of the dataset to avoid bias towards common conditions.
- Used BCEWithLogitsLoss for multilabel classification, with standard training tricks like dropout, batch norm, and learning rate scheduling.
- Evaluated with standard multilabel metrics: exact match ratio, Hamming loss, macro and micro precision/recall/F1.

The results showed that combining image and text embeddings helped, but not by a magic jump—more like a solid nudge. The weighted sum approach gave the model some flexibility to decide what mattered more per feature dimension: this resulted in an overall higher **precision**. The effect of the two model flavors was more pronounced with smaller sample sizes.  

#### Final Thoughts

This project was less about reinventing the wheel and more about creatively exploiting what a powerful pretrained model like CLIP can offer for medical image classification.

Squeezing features out of a black box model, assembling clever prompt ensembles, and combining modalities in simple but effective ways—that’s the essence here. The pathologies were tricky, the data imperfect, and yet the models managed to pick up signals. Not bad for a side hustle with some neural networks.

In short: I didn’t cure disease, but I learned how to make CLIP sweat for its keep.  
