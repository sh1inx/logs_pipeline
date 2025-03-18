import pandas as pd

data = {
    'expression': ['1 + 1', '2 + 2', '3 + 3', '4 + 4', '5 + 5', '10 + 10', '15 + 15'],
    'result': [2, 4, 6, 8, 10, 20, 30] 
}

df = pd.DataFrame(data)

df.to_csv('data/dataset.csv', index=False)
