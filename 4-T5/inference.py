from fine_tune_T5 import PapersSummaryDataModel
from transformers import ( T5TokenizerFast )
import sys
import pandas as pd

MODEL_NAME = 't5-base'
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

def summarizeText(trained_model , text):

    text_encoding = tokenizer(
        text,
        max_length=2000,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding['input_ids'].cuda(),
        attention_mask=text_encoding['attention_mask'].cuda(),
        max_length=512,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )

    preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
    ]
    return "".join(preds)

def main(checkpoint , test_data, output_path):

    trained_model = PapersSummaryDataModel.load_from_checkpoint( checkpoint )
    trained_model.freeze()

    df_test = pd.read_excel(test_data)
    df_evaluate = df_test.copy()

    model_summaries = []
    for index, row in df_test.iterrows():
        text = row['text']  # Assuming 'text' is the column containing your input text

        # Generate a model summary for the text
        model_summary = summarizeText(trained_model, text)

        # Append the model summary to the list
        model_summaries.append(model_summary)

    # Add the model summaries as a new column to the DataFrame
    df_evaluate['model_summary'] = model_summaries

    # Save the updated DataFrame to a new Excel file
    df_evaluate.to_excel(output_path, index=False)

if __name__ == '__main__':
    
    chkpoint_path = sys.argv[1]  ## checkppoint file path
    test_data_path = sys.argv[2]  ## test data path
    output_path = sys.argv[3] ## output path
    main(chkpoint_path, test_data_path, output_path) 
