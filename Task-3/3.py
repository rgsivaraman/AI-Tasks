import pandas as pd

with open("Engineering_graduate_salary.csv") as f:
    data = f.read()
    data=data.split("\n")
    
print(data)
newData = []
for line in data:
    print(line)
    newData.append(line.split(","))

pd.Dataframe(newData,coloumns=["C1","C2","C3","C4","Type"])
df.to_csv('Py.csv)