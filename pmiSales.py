import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
__main__ = '__main__'


#starting point for the program
if __main__ == '__main__':
    p = PMI()
    #p = PMI()
   # p.identifyProductType()
    
class PMI :

    def __init__(self):
        self.df = pd.read_csv('C:/Users/ShresthaAl/Documents/Datasets/converted-pmiSales.csv',
                     encoding = "ISO-8859-1",
                     sep=',')
        self.salesref = pd.read_csv('C:/Users/ShresthaAl/Documents/Datasets/converted-pmiSales-source-reference.csv',
                     encoding = "ISO-8859-1",
                     sep=',')
        #self.productTypes = self.quantify(self.df['Product Type'])
       # self.clients = self.quantify(self.df['Client'])
       # self.insurerProducts = self.quantify(self.df['Insurer Product'])
       # self.insurers = self.quantify(self.df['Insurer'])
       # self.businessSourceWTW = []
        
    @classmethod
    def histogramTransactionYears(self):
        """
            Displays the amount of transaction done through time
        
        """
    
        self.df['Transaction Date'] = self.pd.to_datetime(self.df['Transaction Date'], format='%d/%m/%Y')
    
        self.df['Year'] = self.df['Transaction Date'].dt.year
        self.df['Month'] = self.df['Transaction Date'].dt.month
        self.df['Day'] = self.df['Transaction Date'].dt.day
    
        print(self.df['Year'])
        self.df['Year'].hist()
        #df['Year'].plot.scatter(x=')
    
    def identifyNewSourceOfBusiness(self):
        """
            This method separates the first portion of the data from the Business Source column and
            stores it in an array called businessSourceWTW
            
            In businessSource,
            0 is New
            1 is Renewal
            2 is Pending
            None is ""(blank)
        
        """
        rawBusinessSource = self.df['Business Source']
        businessSource = rawBusinessSource.tolist()
        
        salesRefBsList = self.salesref['Source of Business'].tolist()  #sales reference business source list
        salesRefWtwList = self.salesref['WTW'].tolist()  #sales reference business WTW equivalent
        
        for sourceIndex in range(0,len(businessSource)):
            if businessSource[sourceIndex] in salesRefBsList:
                index = salesRefBsList.index(businessSource[sourceIndex])
                businessSource[sourceIndex] = self.changeToIndex(salesRefWtwList[index])
        
    def quantify(self, passedData):
        """
        Quantifies the data it is passed as a parameter.
        
        """
        newList=[]
        for data in passedData:
            if not data in newList:
                newList.append(data)
        return newList
    def changeToIndex(self, item):
        if(item=='New'):
            return 0
        elif(item=='Renewal'):
            return 1
        elif(item=='Pending'):
            return 2
        else:
            return None
        
    def scatterPlot(self):
        colsToDisplay = ['Product Type', 'Client', 'Insurer Product', 'Business Source', 'Insurer']
        t = zip(self.productTypes,self.clients,self.insurerProducts,self.businessSourceWTW,self.insurers)
        print(t)
        sns.pairplot(self.df[colsToDisplay], size=2.0)
        plt.tight_layout()
        plt.show()

        