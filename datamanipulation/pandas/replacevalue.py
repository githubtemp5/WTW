import pandas as pd

df = pd.read_csv("C:\\Users\\ChoudhuryMB\\Desktop\\keras-work\\industry_dataset17.csv",
                 header=None,
                 sep=',')

df.columns = ['year', 'insurer_product',
              'month', 'industry', 'business_type_new', '"business_type_renewal']





"""
def main(df):
    industry_list = []
    
    for industry in df['industry']:
        if not industry in industry_list:
            industry_list.append(industry)
    
    print("Total length:", len(industry_list))
    
    print(industry_list)
    
main(df)

"""

def main(df):
    industry_list = ['Publishing and Media related activities', 'Financial Services and related industries', 
                     'Industrial Manufacturing and related', 'Other Manufacturing', 
                     'Manufacturing of food and related', 'Public Services Education and related', 
                     'Wholesalers', 'Technical Services', 'Real Estate and Letting Services', 
                     'Retail', 'Miscellaneous', 'Transport and related industries', 'Rental', 
                     'Motor Vechicle related', 'Sporting related activities', 
                     'Hi Tech, Legal, Accountancy and related industries', 
                     'Entertainment and related industries', 'Hotels and Catering', 
                     'Religious Activities', 'Construction and other related', 'Waste Treatment', 
                     'Quarrying. Mining and related industries', 'Farming and Animal related services', 
                     'News Agencies, Libraries and Museums', 'Recycling, Environmental Resources and related', 
                     'Household activities', 'Agent related', 'Management and Holding Companies']









"""
def main(df):
    for year in ["97", "98", "99", "00", "01", "02",
                 "03", "04", "05", "06", "07", "08",
                 "09", "10", "11", "12", "13", "14",
                 "15", "16", "17", "18",]:
        for month in ["Jan", "Feb", "Mar", "Apr", "May",
                      "Jun", "Jul", "Aug", "Sep", "Oct",
                      "Nov", "Dec"]:
            df.loc[df['contract_year']==(month+'-'+year), 'contract_year'] = month
"""

"""
def main(df):
    place_one = 1
    place_zero = 0
    
    source_business = ["New", "Renewal"]
    
    for source in ["New", "Renewal"]:
        if(source == 'New'):
            df.loc[df['business_type']==source, "business_type_new"] = place_one
            df.loc[df['business_type']==source_business[1], "business_type_new"] = place_zero 
        elif(source=='Renewal'):
            df.loc[df['business_type']==source, "business_type_renewal"] = place_one
            df.loc[df['business_type']==source_business[0], "business_type_renewal"] = place_zero
            
"""
"""

new = 1
renewal = 1

def main(df):
    for source in ["New", "Renewal"]:
        if(source == "New"):
            df.loc[df['business_type']==source, "business_type_new"] = new
        elif(source == "Renewal"):
            df.loc[df['business_type']==source, "business_type_renewal"] = renewal
"""

"""
def main(df):
    df.loc[df['business_type_new']=='0' , "business_type_new"] = "0"
    df.loc[df['business_type_renewal']=='0' , "business_type_renewal"] = "0"
"""
    
"""
def main(df):
    df.loc[df['contract_year']=='Jan', "contract_year"] = 1
    df.loc[df['contract_year']=='Feb', "contract_year"] = 2
    df.loc[df['contract_year']=='Mar', "contract_year"] = 3
    df.loc[df['contract_year']=='Apr', "contract_year"] = 4
    df.loc[df['contract_year']=='May', "contract_year"] = 5
    df.loc[df['contract_year']=='Jun', "contract_year"] = 6
    df.loc[df['contract_year']=='Jul', "contract_year"] = 7
    df.loc[df['contract_year']=='Aug', "contract_year"] = 8
    df.loc[df['contract_year']=='Oct', "contract_year"] = 9
    df.loc[df['contract_year']=='Sep', "contract_year"] = 10
    df.loc[df['contract_year']=='Nov', "contract_year"] = 11
    df.loc[df['contract_year']=='Dec', "contract_year"] = 12
"""

"""
def main(df):
    df.loc[df['industry']==' ' , "industry"] = "NaN"
    df.loc[df['industry']=='0' , "industry"] = "NaN"
"""

"""
def main(df):
    df.dropna(subset=['industry'], inplace=True)
"""



"""
print("Starting to write...")
main(df)
print("Saving...")
df.to_csv("industry_dataset18.csv")
print("Created new file")
"""
