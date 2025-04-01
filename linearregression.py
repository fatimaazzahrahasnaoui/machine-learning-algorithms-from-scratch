import numpy as np 
# les donnes 
x=np.array([1,2,3,4,4,6])
y=np.array([2,5,6,7,9,8.])
# intialiser  the parametres 
w=0 
b=0
learning_rate= 0.01
epochs=1000 # combien de fois lalgo parcourt  data p bach update w et b for descnete 
def mean_sequerd_eror(ytrue,ypredict):
    return np.mean((ytrue-ypredict)**2)
#descente de gradient 
for i in range(epochs):
    ypredict=w*x+b
    eror=mean_sequerd_eror(y,ypredict)
    wgradient=-2*np.dot(x,y-ypredict)/len(x)
    bgradient=-2*np.mean(y-ypredict)
    w-=learning_rate*wgradient
    b-=learning_rate*bgradient
    if i%100==0:
        print(f"epoch{i}: w={w},b={b} , error={eror}")
print("Valeurs finales: w =", w, "b =", b)
