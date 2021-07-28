# Transfer Learning across Transformer Models on Multi Language Translation Task 

Kevin Ngo, UC Berkeley MIDS Program (kngo@berkeley.edu)\
Prakhar Maini, UC Berkeley MIDS Program (prakharmaini@berkeley.edu)

With many thanks to our advisors: Joachim Rahmfeld and Drew Plant for their guidance on this research.


## Paper

Please read our paper: 
[Transfer Learning across Transformer Models on Multi Language Translation Task ](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/TL-MMT.pdf).

## Major Pre-requisites

- torch, torchvision (PyTorch) 1.9 or higher

- transformers (Hugging Face) 4.8 or higher


## Instructions

- To obtain the pretrained model, base tokenizer, and dataset, download the links and extract the files within the cloned repo. Please go to [Facebook's Large Scale Multilingual Translation Task](http://www.statmt.org/wmt21/large-scale-multilingual-translation-task.html) for full details.
    * [Pretrained Model Small](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz)
    * [Pretrained Model Large](https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz)
    * [Training Dataset (Task1/Main)](http://data.statmt.org/wmt21/multilingual-task/small_task1_filt_v2.tar.gz)
    * [Training Dataset (Task2)](http://data.statmt.org/wmt21/multilingual-task/small_task2_filt_v2.tar.gz). This is only required to run [analysis_dataset_size](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/analysis_dataset_size.ipynb).
    * [Validation/Test Dataset](https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz)


- To obtain the values from tables 1 in the paper, run the [dataset_creation](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/dataset_creation.ipynb) notebook.

- To recreate the results of the paper,<br>
    1. To Create the dataset, runn the [analysis_dataset_size](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/analysis_dataset_size.ipynb) notebook.

    2. To Create the model, train the model, and generate the test BLEU score, run the [pipeline](https://github.com/Ngo-Kevin/Transfer-Learning-Across-Transformer/blob/main/pipeline.ipynb) notebook with adjusted settings.

## Hardware and run-time expectations

Authors used NVIDIA Tesla V100 graphics cards on various GCP instances for training. Training with the largest dataset size of 120K examples with a batch size of 32 for 30 epoch took ~20 hours.