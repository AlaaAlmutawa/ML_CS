
## README

Dataset can be found in: https://huggingface.co/datasets/scientific_papers

## Decoder-only model: 

1. Tokenization: 
- tokenize_dataset.py: script to use sentence piece tokeneization on a given dataset. 

Run the command: 

```
python tokenize_dataset.py --input '[FILE].xlsx' --output_model '[MODEL_NAME]' --output_txt '[NAME_OF_TEXT_OUTPUT].txt'
```

Place the output model to vocab_dir 

2. Training model: 

- simple_gpt.py: scrtipt to build the model, and train the model using the specified configurations. 
- simple_gpt_model.py: script that includes all the method hat builds the model archiecture and has the training loop definition. 
- utils_v1.py: script that has some helper functions. 

Run the command: 

```
python simple_gpt.py
```

3. Inference: 

- Inference_on_decoder_only_model.ipynb: notebook that was used on colab to run inference on the trained model. 


## LLama 2: 

1. Finetuning: 

- Truncation_of_article_to_1024_Text_Summarization_Using_LLama2.ipynb: finetuning opensource llama 2 on our dataset \
notes: that the dataset should be present in the same directory. <br> you also need to specify your huggingface login token to be able to load the pretrained model. 

2. Inference: 

- Inference_of_llama2_finetuned_model.ipynb: This notebook is used to use the finetuned llama2 model for inference <br> In the notebook, you can find: <br> 1. merging of the base model with the finetuned weights from QLoRA adapters <br> 2. retrieving the finetuned model from huggingface repository.<br> 3. using the finetuned model for inference. 


## T5:
 
FineTune:
 
```
python fine_tune_T5.py <input_file_path>
```
The input file should have following columns.
1. article : the text to be summarized
2. summary : the reference summary
 
Inference:
 
```
python inference.py <checkpoint_path> <test_data_path> <output_file_path>
```
The test data file should have the text to be summarized as `text`.
 
## Evaluation:
 
```
python evaluate.py <input_file_path> <output_file_path>
```
The input file should have the following columns.
1. summary : reference summary
2. model_summary : model generated summary


## References: 
Below are the references used in this project. Thanks to the help of tutorials and youtube contributors we were able to explore different transfomer models in our project. 

Decoder-only: 
1. Coursera Deep Learning specialization: https://www.coursera.org/specializations/natural-language-processing
2. LaurentVeyssier's Github repository: https://github.com/LaurentVeyssier/TRAX_transformer_abstractive_summarization_model/blob/main/TRAX_transformer_summarizer_model.ipynb

Finetuning LLama 2 model: 
1. tutorial on youtube: https://www.youtube.com/watch?v=MDA3LUKNl1E
2. Curiousily's github repository: https://github.com/curiousily/Get-Things-Done-with-Prompt-Engineering-and-LangChain/blob/master/14.fine-tuning-llama-2-7b-on-custom-dataset.ipynb
3. kaggle tutorial: https://www.kaggle.com/code/mahimairaja/fine-tuning-llama-2-tweet-summarization

Finetuning T5 model: 
1. tutorial on youtube: https://www.youtube.com/watch?v=KMyZUIraHio&t=180s
2. animesh-algorithm's github repository: https://github.com/animesh-algorithm/Text-Summarization-using-T5-transformers-and-Pytorch-Lightning/blob/main/Text_Summarization_Using_Transformer_T5_and_Pytorch_Lightning.ipynb

