import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
import numpy as np 
# les donnes 
x=np.array([1,2,3,4,4,6])
y=np.array([0,1,0,0,1,1])
# intialiser  the parametres 
w=0 
b=0
learning_rate= 0.01
epochs=1000 # combien de fois lalgo parcourt  data p bach update w et b for descnete 

#descente de gradient 
for i in range(epochs):
    linearmodel=w*x+b
    ypredict=sigmoid(linearmodel)
    #Entropy Cross (or Cross-Entropy Loss) is a common cost function used in classification problems, particularly in binary classification tasks with logistic regression. It measures the difference between the predicted probability distribution (from the model) and the actual distribution (the true labels).

    #Understanding Cross-Entropy Loss
    #Purpose: The purpose of cross-entropy is to quantify how well the predicted probabilities match the actual labels. If the predicted probabilities are close to the actual labels, the cross-entropy loss will be low. If they are far apart, the loss will be high.
    cost = -np.mean(y * np.log(ypredict) + (1 - y) * np.log(1 - ypredict))
    wgradient=-2*np.dot(x,y-ypredict)/len(x)
    bgradient=-2*np.mean(y-ypredict)
    w-=learning_rate*wgradient
    b-=learning_rate*bgradient
    if i%100==0:
        print(f"epoch{i}: w={w},b={b}")
print("Valeurs finales: w =", w, "b =", b)
