import pandas as pd

df = pd.read_csv("C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\data\\PMI_dataset_large.csv",
                 sep=',')

def main(df):
    
    df.loc[df['Months']=='Jan', "Months"] = 1
    df.loc[df['Months']=='Feb', "Months"] = 2
    df.loc[df['Months']=='Mar', "Months"] = 3
    df.loc[df['Months']=='Apr', "Months"] = 4
    df.loc[df['Months']=='May', "Months"] = 5
    df.loc[df['Months']=='Jun', "Months"] = 6
    df.loc[df['Months']=='Jul', "Months"] = 7
    df.loc[df['Months']=='Aug', "Months"] = 8
    df.loc[df['Months']=='Oct', "Months"] = 9
    df.loc[df['Months']=='Sep', "Months"] = 10
    df.loc[df['Months']=='Nov', "Months"] = 11
    df.loc[df['Months']=='Dec', "Months"] = 12
    
print("Starting to write...")
main(df)
print("Saving...")
df.to_csv("test4.csv")
print("Created new file")
