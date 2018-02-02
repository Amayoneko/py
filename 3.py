import numpy as np  
  
import matplotlib  
matplotlib.use('Agg')  
  
from matplotlib.pyplot import plot,savefig  
import matplotlib.pyplot as plt
  
x=np.linspace(-4,4,30)  
y=np.sin(x);  
  
plot(x,y,'--*b')


