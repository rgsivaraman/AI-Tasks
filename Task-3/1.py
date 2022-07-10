import numpy as np
np.random.seed(100) 
a = np.random.uniform(1,50, 20) 
print(a)
b = np.where(a>45)
c = np.where(a<15)
e=np.clip(a,15,45)
print(e)

