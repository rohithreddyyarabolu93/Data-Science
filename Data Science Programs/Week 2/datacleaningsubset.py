import pandas as pd
df=pd.DataFrame({'ID':[1,2,2,3,4,4,5],
    'Name':['Alice','Bob','Bob','Charlie','David','David','David'],
    'Age':[25,30,30,35,40,40,40]})
print("Original Dataset:\n",df)
df_subset=df.drop_duplicates(subset=['ID'])
print("\nAfter Subset Match Removal (ID):\n",df_subset)