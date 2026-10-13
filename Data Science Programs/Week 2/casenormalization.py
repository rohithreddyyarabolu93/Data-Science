import pandas as pd
df=pd.DataFrame({'Name':['Alice','BOB','charlie','DAVID']})
print("Original Data:\n",df)

df['Name_lower']=df['Name'].str.lower()
df['Name_upper']=df['Name'].str.upper()
print("\n",df)