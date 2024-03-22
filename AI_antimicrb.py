pip install pandasai
pip install polars
from pandasai.llm import OpenAI
llm = OpenAI(api_token="..-N.........kFJufD4EUJ8....LkF")
import pandas as pd
from pandasai import SmartDataframe
data1= pd.read_csv(r"C:\...\...\Documents\Venatorx surveillance data for Vivli 27Feb2023.csv")
data1.head()
df2 = SmartDataframe(data1, config={"llm": llm})

df2.chat('what is the predicted FEP_MIC for Serratia ureilytica by Country and bodysite?')
df2.chat('plot top 5 FEP_MIC for Serratia ureilytica by Country using different colors')

data2= pd.read_csv(r"C:\...\...\Documents\Omadacycline_2014_to_2022_Surveillance_data.csv")
data2.head()

df3 = SmartDataframe(data2, config={"llm": llm})

df3 = SmartDataframe(data2, config={"llm": llm})
df3.chat('scatter plot  for amikacin by  infection type and nosocomial with different colors and slanted  and shortened x labels ,y axis label ascending, legends outside plot area')
pip install scikit-learn
pip install seaborn
df3.chat('plot the regression analysis for amikacin by  infection type and nosocomial')
