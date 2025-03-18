import pandas as pd

data = {
    'expression': ['1 + 1', '2 + 2', '3 + 3', '4 + 4', '5 + 5'],
    'result': [2, 4, 6, 8, 10] 
}

df = pd.DataFrame(data)

df.to_csv('data/dataset.csv', index=False)
