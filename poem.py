import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Activation
from tensorflow.keras.optimizers import RMSprop
#Loading the text file form the link
filepath= tf.keras.utils.get_file("shakespeare.txt","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
#loading the file for processing 
text= open(filepath,"rb").read().decode(encoding="utf-8").lower()

characters=sorted(set(text))

#Creating a dictionary to access characters  
char_to_index= dict((c,i) for i,c in enumerate(characters))
index_to_char= dict((i,c) for i,c in enumerate(characters))
#Initializing the sentence length and skip sequence
seq_len= 40
step_size=3
sentences=[]
next_char= []

#converting the text in number for the model training 
for i in range(0,len(text)-seq_len,step_size):
    sentences.append(text[i : i+seq_len])
    next_char.append(text[i+seq_len])
    

x= np.zeros((len(sentences),seq_len,len(characters)),dtype=np.bool)
y= np.zeros((len(sentences),len(characters)),dtype=np.bool)

for i , sentence in enumerate(sentences):
    for t , charc in enumerate(sentence):
        x[i,t,char_to_index[charc]]=1
    y[i,char_to_index[next_char[i]]]=1
    
# building the neural network 
model=Sequential()
model.add(LSTM(128,input_shape=(seq_len,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",optimizer=RMSprop(lr=0.01))

model.fit(x,y,batch_size=256,epochs=4)

#function to predict the next character
def sample(preads,temp=1.0):
    preads=np.asarray(preads).astype("float64")
    preads=np.log(preads)/temp
    ex_preads=np.exp(preads)
    preads=ex_preads/np.sum(ex_preads)
    prob=np.random.multinomial(1,preads,1)
    return np.argmax(prob)

#function to generate the text
def generate_text(lean,temp):
    start=random.randint(0,len(text)-seq_len-1)
    genrate=""
    sen=text[start:start+seq_len]
    genrate += sen
    for i in range(lean):
        x=np.zeros((1,seq_len,len(characters)))
        for t , char in enumerate(sen):
            x[0,t,char_to_index[char]]=1
        pred=model.predict(x,verbose=0)[0]
        next_ind=sample(pred,temp)
        next_char=index_to_char[next_ind]
        genrate += next_char
        sen=sen[1:]+next_char
    return genrate
#pass the length of the sentence and the temperature 
print(generate_text(300,0.5))