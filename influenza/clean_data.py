import pandas as pd

df = pd.read_csv('NewMexico_FluSurveillance.csv')

df.columns = df.columns.str.strip()

df_filtered = df[(df['SEX CATEGORY'] == 'Overall') & (df['RACE CATEGORY'] == 'Overall')]
df_filtered = df_filtered.drop(columns=["SEX CATEGORY", "RACE CATEGORY", "CUMULATIVE RATE", "AGE ADJUSTED CUMULATIVE RATE", "AGE ADJUSTED WEEKLY RATE"])

print(df_filtered.head(22))