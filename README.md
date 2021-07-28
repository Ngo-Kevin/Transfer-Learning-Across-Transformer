# Transfer Learning across Transformer Models on Multi Language Translation Task 

Kevin Ngo, UC Berkeley MIDS Program (kngo@berkeley.edu)\
Prakhar Maini, UC Berkeley MIDS Program (prakharmaini@berkeley.edu)

With many thanks to our advisors: Joachim Rahmfeld and Drew Plant for their guidance on this research.

### **Paper:** [Transfer Learning across Transformer Models on Multi Language Translation Task ](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/TL-MMT.pdf).

<br>

## Abstract

Recent work in translation 1, 2 has focused on creating large-scale multilingual transformer models capable of achieving state-of-the-art performance. These mo- dels attempt to leverage the shared structure of multiple languages to improve down-stream task per- formance in low-resource languages. In this work, we explore the potential of transfer learning between transformer models using two recently released multilingual transformer models (M2M-100 1, T5 3) by utilizing the M2M-100’s embeddings within the T5’s architecture in order to expand T5’s translation capabilities to low resource languages such as Esto- nian and Macedonian. Our experiments suggest that usage of T5’s pretrained encoder & decoder weights improves model performance on previously untrained low-resource language translation even with small training dataset compared to random initialization of the weights. Augmenting the training with pre-learnt multilingual embeddings improves the performance further but this improvement is only seen with larger training set sizes. 

<br>

## Major Requirement
- torch, torchvision (PyTorch) 1.9 or higher

- transformers (Hugging Face) 4.8 or higher

<br>

## Instructions
- To obtain the pretrained model, base tokenizer, and dataset, download the links and extract the files within the cloned repo. Please go to [Facebook's Large Scale Multilingual Translation Task](http://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html) for full details.
    * [Pretrained Model-Small](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz)
    * [Pretrained Model-Large](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz)
    * [Training Dataset (Task1/Main)](http://data.statmt.org/wmt21/multilingual-task/small_task1_filt_v2.tar.gz)
    * [Training Dataset (Task2)](http://data.statmt.org/wmt21/multilingual-task/small_task2_filt_v2.tar.gz). This is only required to run [analysis_dataset_size](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/analysis_dataset_size.ipynb).
    * [Validation/Test Dataset](https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz)


- To recreate the results of the paper,<br>
    - To convert all individual files into a dataset, run the [dataset_creation](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/dataset_creation.ipynb) notebook.
    - To obtain the values from tables 1 in the paper, run the [analysis_dataset_size](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/analysis_dataset_size.ipynb) notebook.
    - To create Figure 2 and Figure 3 in the paper, run the [analysis_dataset_tokenized_size](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/analysis_dataset_tokenized_size.ipynb) notebook.
    - To create the model, train the model, and generate the test BLEU score, run the [pipeline](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/pipeline.ipynb) notebook with adjusted settings.

<br>

## Hardware and run-time expectations
Authors used NVIDIA Tesla V100 graphics cards on various cloud instances for training. Training the model with 120K examples in a batch size of 32 for 30 epoch took ~20 hours.
