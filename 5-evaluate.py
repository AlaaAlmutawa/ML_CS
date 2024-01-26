import pandas as pd
from evaluate import load
import sys

def save_scores_to_excel(title,df):
    df.to_excel(f'{title}_rouge_scores.xlsx', index=False)


path = sys.argv[1]  ## Input excel file with outputs
output_file = sys.argv[2] ## output file name
df = pd.read_excel(path)  

rouge = load('rouge')

results = rouge.compute(predictions=df['model_summary'], references=df['summary'],use_aggregator=False)
df_result = pd.DataFrame(results)
results2 = rouge.compute(predictions=df['model_summary'], references=df['summary'])

concatenated_df = pd.concat([df, df_result ], axis=1)

# Save scores to a DataFrame and Excel file
save_scores_to_excel(f'{output_file}',concatenated_df) 
save_scores_to_excel(f'{output_file}_summary',results2)