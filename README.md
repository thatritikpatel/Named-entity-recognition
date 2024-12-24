# Named Entity Recognition (NER) with BERT

## Overview

Named Entity Recognition (NER) is a sub-task of information extraction in natural language processing (NLP). It involves identifying entities in text and classifying them into predefined categories such as persons, locations, organizations, or miscellaneous entities. This project demonstrates how to fine-tune the pre-trained BERT ("bert-base-uncased") model using the CoNLL-2003 dataset to perform NER.

### What is Named Entity Recognition (NER)?
Named Entity Recognition (NER) is the process of locating and classifying named entities in unstructured text into predefined categories. These entities can include:
- **Persons**: Names of people.
- **Locations**: Geographic locations such as cities, states, or countries.
- **Organizations**: Institutions, companies, or groups.
- **Miscellaneous Entities**: Other entities that do not fall into the above categories.

### Benefits of NER
- **Efficient Information Retrieval**: Helps extract critical information from large text corpora.
- **Improved Text Understanding**: Facilitates better semantic understanding of the text.
- **Automation of Repetitive Tasks**: Enables automatic tagging of documents and categorization.
- **Foundation for Advanced NLP Tasks**: Provides essential preprocessing for tasks like question answering and document summarization.

### Challenges Without NER
- **Manual Effort**: Extracting specific entities manually from large datasets is time-consuming and error-prone.
- **Data Overload**: Without NER, identifying relevant information in unstructured text is difficult.
- **Inefficiency in Downstream Applications**: NLP applications like chatbots or search engines may perform poorly without structured information.

### Practical Uses of NER
- **Healthcare**: Extracting medical terms, symptoms, and drug names from clinical records.
- **Finance**: Identifying company names, financial terms, and market events in reports.
- **Customer Support**: Analyzing user feedback to identify specific issues or product mentions.
- **Legal**: Extracting case references, legal entities, or contract terms from documents.
- **News Analysis**: Categorizing and summarizing news articles by identifying key persons, locations, or events.

## Project Details

### Dataset: CoNLL-2003

The CoNLL-2003 dataset is a widely used benchmark for NER tasks. It includes annotations for four types of named entities: **persons**, **locations**, **organizations**, and **miscellaneous entities**. The dataset is structured in an IOB2 tagging scheme and contains splits for training, validation, and testing.

- **Dataset URL**:
  - [Hugging Face Dataset](https://huggingface.co/datasets/eriktks/conll2003)
  - [Official Dataset Website](https://www.clips.uantwerpen.be/conll2003/ner/)
  - [Publication](https://www.aclweb.org/anthology/W03-0419)

### Tools and Libraries
This project uses the following libraries:
- **Transformers**: For accessing the pre-trained BERT model.
- **Datasets**: For loading and managing the CoNLL-2003 dataset.
- **Tokenizers**: For tokenizing the text data.
- **Seqeval**: For evaluation of NER models.
- **Evaluate**: For calculating precision, recall, and F1 scores.

### Key Components of the Project

#### 1. Pre-trained Model
We use the "bert-base-uncased" model from Hugging Face. Pre-trained models like BERT provide a strong starting point for NLP tasks by leveraging representations learned from large text corpora.

#### 2. Fine-tuning
Fine-tuning is the process of taking a pre-trained model and training it further on a specific task or dataset. Here, we fine-tune BERT on the CoNLL-2003 dataset for the NER task.

**Importance of Fine-tuning:**
- Allows the model to adapt to task-specific data.
- Reduces the need for large labeled datasets by building on pre-trained knowledge.
- Improves task performance compared to training a model from scratch.

**Challenges in Fine-tuning:**
- Computationally intensive, requiring significant resources.
- Risk of overfitting if the dataset is small.
- Requires careful hyperparameter tuning.

**Benefits of Fine-tuning:**
- Leverages the contextual understanding of language from pre-training.
- Produces state-of-the-art results on many NLP tasks.
- Saves time and effort compared to training models from scratch.

#### 3. Model Architecture
The fine-tuned BERT model is configured for token classification. Each token is classified into one of the entity labels defined in the CoNLL-2003 dataset.

#### 4. Tokenization
We use the `BertTokenizerFast` for splitting text into tokens compatible with BERT. Tokenization ensures proper alignment between input text and model requirements.

#### 5. Data Collation
The `DataCollatorForTokenClassification` is used to pad and batch the tokenized data for efficient processing.

#### 6. Training
Training is performed using the Hugging Face Trainer API, which simplifies the training process with features like learning rate scheduling, gradient accumulation, and evaluation during training.

#### 7. Evaluation
The model is evaluated using:
- **Precision**: The proportion of correctly predicted entities out of all predicted entities.
- **Recall**: The proportion of correctly predicted entities out of all true entities.
- **F1 Score**: The harmonic mean of precision and recall.

Evaluation metrics are calculated using the Seqeval library.

### Implementation
Key Python imports and steps include:

```python
import datasets
import numpy as np
from transformers import BertTokenizerFast, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification, TrainingArguments, Trainer
import evaluate

# Load the dataset
from datasets import load_dataset
conll2003 = load_dataset("conll2003")

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=conll2003["train"],
    eval_dataset=conll2003["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)

# Train and evaluate
trainer.train()
```

### Challenges Faced
- **Data Preprocessing**: Aligning tokenized text with entity labels is non-trivial.
- **Resource Limitations**: Fine-tuning large models like BERT requires considerable computational resources.
- **Evaluation**: Handling the subtleties of IOB tagging during evaluation is challenging.

### Benefits of this Approach
- Achieves high accuracy and F1 scores on the NER task.
- Demonstrates transfer learning's effectiveness in NLP.
- Provides a scalable approach to other NER datasets or tasks.

## Conclusion
This project showcases the power of fine-tuning pre-trained models like BERT for NER tasks. By leveraging the CoNLL-2003 dataset and state-of-the-art libraries, it highlights both the challenges and benefits of implementing a robust NER pipeline. The code and methodology can be extended to other NLP tasks with minimal modifications.

## Contact
- Ritik Patel - [ritik.patel129@gmail.com]