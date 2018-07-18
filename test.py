df = pd.read_csv('C:\\Users\ChoudhuryMB\\Documents\\ds\\converted-data.csv',
                     encoding = "ISO-8859-1",
                     sep=',',
                     error_bad_lines=False,
                     index_col=False,
                     dtype='unicode')

df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%d-%m-%Y')

df['Year'] = df['Transaction Date'].dt.year
df['Month'] = df['Transaction Date'].dt.month
df['Day'] = df['Transaction Date'].dt.day

print(df['Year'])
