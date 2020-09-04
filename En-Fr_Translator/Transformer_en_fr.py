# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:20:34 2020

@author: Javed
"""

'''
Part 1: Importing dependencies and loading input data.
The data has been downlowaded from 'https://www.statmt.org/europarl/'

P85-Non-Breaking-Prefix.en and P85-Non-Breaking-Prefix.fr are some of the
stopwords which don't come along the data. These are manually created and
I have added that to the data folder. Feel free to change these as you seem
appropriate to tune the model.
'''
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers
import time
import tensorflow_datasets as tfds

## Reading the files

start_time=time.time()
with open('europarl-v7.fr-en.en',mode='r',encoding='utf-8') as f:
    europarl_en=f.read()

with open('europarl-v7.fr-en.fr',mode='r',encoding='utf-8') as f:
    europarl_fr=f.read()
    
with open('P85-Non-Breaking-Prefix.en',mode='r',encoding='utf-8') as f:
    Non_Breaking_en=f.read()
    
with open('P85-Non-Breaking-Prefix.fr',mode='r',encoding='utf-8') as f:
    Non_Breaking_fr=f.read()
print("Time taken to load the data: ",time.time()-start_time)



'''
Part 2: Cleaning and transforming data
'''
#type(europarl_en)

Non_Breaking_en=Non_Breaking_en.replace('\n',' ').split()
Non_Breaking_fr=Non_Breaking_fr.replace('\n',' ').split()

### Cleaning the data

corpus_en= europarl_en

for x in Non_Breaking_en:
    corpus_en.replace(x,x+'###')
##If a stop is not followed by a space we are not considering that as sentence end
corpus_en=re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])",".###",corpus_en)
corpus_en=re.sub(r".###",'',corpus_en)
corpus_en=re.sub(r" +",' ',corpus_en)
corpus_en=corpus_en.split('\n')



corpus_fr= europarl_fr

for x in Non_Breaking_fr:
    corpus_fr.replace(x,x+'###')
                      
corpus_fr=re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])",".###",corpus_fr)
corpus_fr=re.sub(r".###",'',corpus_fr)
corpus_fr=re.sub(r" +",' ',corpus_fr)
corpus_fr=corpus_fr.split('\n')

    
 ### Tokenizing the corpus
 
tokenizer_en=tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_en,target_vocab_size=2**13)
tokenizer_fr=tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_fr,target_vocab_size=2**13)

vocab_size_en=tokenizer_en.vocab_size + 2
vocab_size_fr=tokenizer_fr.vocab_size + 2


## Adding an extra token each at the begining and end of each sentence:

inputs=[[vocab_size_en-2]+tokenizer_en.encode(sentence)+[vocab_size_en-1] for sentence in corpus_en]  
outputs=[[vocab_size_fr-2]+tokenizer_fr.encode(sentence)+[vocab_size_fr-1] for sentence in corpus_fr]
'''
I've are just considering a size of length 20 in our analysis due to technical 
limitations and high computation requirements. But I've got decent results with
it. Feel free to play around it.
'''
## removing the long sentences

max_length=20
idx_to_remove=[count for count, sent in enumerate(inputs) if len(sent)>max_length]

for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]

idx_to_remove=[count for count, sent in enumerate(outputs) if len(sent)>max_length]    
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]
    
## Padding the inputs
    
inputs=tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                     value=0,
                                                     padding='post',
                                                     maxlen=max_length)

outputs=tf.keras.preprocessing.sequence.pad_sequences(outputs,
                                                     value=0,
                                                     padding='post',
                                                     maxlen=max_length)


batch_size=64
buffer_size=20000 #For data shuffle

dataset= tf.data.Dataset.from_tensor_slices((inputs,outputs))

dataset=dataset.cache() ##Just for speeding up. Doesn't change anythong else
dataset=dataset.shuffle(buffer_size).batch(batch_size)
dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)

'''
Part 3: Defining functions and creating Model architecture
The architecture is purely based on Google's paper:
    ATTENTION IS ALL YOU NEED
Check this out at: https://arxiv.org/abs/1706.03762
The paper is short and easily understandable.
'''

class PositionalEncoding(layers.Layer):
    
    def __init__(self):
        super(PositionalEncoding,self).__init__()
        
    def get_angles(self,pos,i,d_model): ##pos (seq_length,1) i:(1,d_model)
        
        angles=1/np.power(10000.,(2*(i//2))/np.float32(d_model))
        return pos*angles
    
    def call(self,inputs):
        seq_len=inputs.shape[-2]  ##Check
        d_model=inputs.shape[-1]  ##Check
        
        angles=self.get_angles(np.arange(seq_len)[:,np.newaxis],
                               np.arange(d_model)[np.newaxis,:],
                               d_model)
        
        angles[:,0::2]=np.sin(angles[:,0::2])
        angles[:,1::2]=np.cos(angles[:,1::2])
        
        pos_encoding=angles[np.newaxis,...]  ###... is the ellipses indicates as many as required 
        
        return inputs+tf.cast(pos_encoding,tf.float32)



def scaled_dot_product_attention(querries,keys,values,mask):
    product= tf.matmul(querries,keys,transpose_b=True)
    
    keys_dims=tf.cast(tf.shape(keys)[-1],tf.float32)
    
    scaled_product = product/tf.math.sqrt(keys_dims)
    
    '''
    The mask is a 1 - lower traingle matrix. Masking ensures that the
predictions for position i can depend only on the known outputs at positions less than i.

Since we apply a softmax after the masking, we will be multipliying a large -ve number
so that the result of the softmax is zero.
    '''
    if mask is not None:
        scaled_product=scaled_product+(mask * -1e9)
        
    attention= tf.matmul(tf.nn.softmax(scaled_product,axis=-1),values)
    
    return attention


class MultiHeadAttention(layers.Layer):
    
    def __init__(self,nb_proj):
        super(MultiHeadAttention,self).__init__()
        self.nb_proj=nb_proj
        
        
    def build(self,input_shape=(20,)):
        
        self.d_model=input_shape[-1]
        assert self.d_model % self.nb_proj == 0
        
        self.d_proj=self.d_model//self.nb_proj
        
        self.query_lin=layers.Dense(units=self.d_model)
        self.key_lin=layers.Dense(units=self.d_model)
        self.value_lin=layers.Dense(units=self.d_model)        
        self.fin_lin=layers.Dense(units=self.d_model)
        
    
    def split_proj(self,inputs,batch_size):  ## inputs: (batch_size, seq_lenght,d_model)
        
        shape=(batch_size,-1,self.nb_proj,self.d_proj)  ##What is the need for -1(instead of seq_len)?
        
        splitted_inputs= tf.reshape(inputs,shape)  ## (batch_size,seq_legth,no_proj,d_model)
        
        return tf.transpose(splitted_inputs,perm=[0,2,1,3]) ## (batch_size, no_proj,seq_legth,d_model)
        
    
    def call(self,querries,keys,values,mask):
        
        ##Preparing the querriers, keys and values
        
        batch_size=tf.shape(querries)[0]
        querries=self.query_lin(querries)
        keys=self.key_lin(keys)
        values=self.value_lin(values)
        
        ##Splitting the querries, keys and values
        querries=self.split_proj(querries,batch_size)
        keys=self.split_proj(keys,batch_size)
        values=self.split_proj(values,batch_size)
        
        attention = scaled_dot_product_attention(querries,keys,values,mask)
        
        attention=tf.transpose(attention,perm=[0,2,1,3])
        concat_attention=tf.reshape(attention,shape=(batch_size,-1,self.d_model))
        
        outputs=self.fin_lin(concat_attention)
        
        return outputs


### Encoder
        
class EncoderLayer(layers.Layer):
    
    def __init__(self,FFN_units,nb_proj,dropout):
        super(EncoderLayer,self).__init__()
        self.FFN_units=FFN_units
        self.nb_proj=nb_proj
        self.dropout=dropout
        
    def build(self,input_shape=(20,)):
        self.d_model=input_shape[-1]
        self.multi_head_attention=MultiHeadAttention(self.nb_proj)
        self.dropout_1=layers.Dropout(rate=self.dropout)
        self.norm_1=layers.LayerNormalization(epsilon=1e-6)
        self.dense_1=layers.Dense(units=self.FFN_units,activation='relu')
        ##We want to keep the dimension same as the input because this will be fed to the decoder and  facilitate residual connections
        self.dense_2=layers.Dense(units=self.d_model) 
        self.dropout_2=layers.Dropout(rate=self.dropout)
        self.norm_2=layers.LayerNormalization(epsilon=1e-6)
        
    def call(self,inputs,mask,training):
        attention=self.multi_head_attention(inputs,
                                            inputs,
                                            inputs,
                                            mask)
        attention=self.dropout_1(attention,training=training)
        attention=self.norm_1(attention+inputs)
        outputs=self.dense_1(attention)
        outputs=self.dense_2(outputs)
        outputs=self.norm_2(outputs+attention)
        
        return outputs
    
    
class Encoder(layers.Layer):
    
    def __init__(self,nb_layers,FFN_units,nb_proj,dropout,vocab_size,d_model,name='encoder'):
        super(Encoder,self).__init__(name=name)
        self.nb_layers=nb_layers
        self.FFN_units=FFN_units
        self.nb_proj=nb_proj
        self.dropout=dropout
        self.vocab_size=vocab_size
        self.d_model=d_model
        
        self.embedding=layers.Embedding(vocab_size,self.d_model)
        self.pos_encoding=PositionalEncoding()
        self.dropout=layers.Dropout(rate=self.dropout)
        self.enc_layers=[EncoderLayer(FFN_units,nb_proj,dropout) for _ in range(nb_layers)]
        
    def call(self,inputs,mask,training):
        outputs=self.embedding(inputs)
        outputs*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        outputs=self.pos_encoding(outputs)
        outputs=self.dropout(outputs,training)
        
        for i in range(self.nb_layers):
            outputs=self.enc_layers[i](outputs,mask,training)
            
        return outputs
    
    
    
class DecoderLayer(layers.Layer):
    
    def __init__(self,FFN_units,nb_proj,dropout):
        super(DecoderLayer,self).__init__()
        self.FFN_units=FFN_units
        self.nb_proj=nb_proj
        self.dropout=dropout
        
        
    def build(self,input_shape=(20,)):
        self.d_model=input_shape[-1]
        self.multi_head_attention=MultiHeadAttention(self.nb_proj)
        self.dropout_1=layers.Dropout(rate=self.dropout)
        self.norm_1=layers.LayerNormalization(epsilon=1e-6)
        self.dense_1=layers.Dense(units=self.FFN_units,activation='relu')
        self.dense_2=layers.Dense(units=self.d_model)
        #self.dropout_2=layers.Dropout(rate=self.dropout)
        #self.norm_2=layers.LayerNormalisation(epsilon=1e-6)
        
    def call(self, inputs, enc_outputs,mask_1,mask_2,training):
        
        attention=self.multi_head_attention(inputs,inputs,inputs,mask_1)
        attention=self.dropout_1(attention,training)
        attention=self.norm_1(attention+inputs)
        
        attention_2 = self.multi_head_attention(attention
                                                ,enc_outputs
                                                ,enc_outputs
                                                ,mask_2)
        
        attention_2=self.dropout_1(attention_2)
        attention_2=self.norm_1(attention_2+attention)
        
        outputs=self.dense_1(attention_2)
        outputs=self.dense_2(outputs)
        outputs=self.dropout_1(outputs)
        outputs=self.norm_1(outputs+attention_2)
        
        return outputs
    
    

class Decoder(layers.Layer):
    
    def __init__(self
                 ,nb_layers
                 ,FFN_units
                 ,nb_proj
                 ,dropout
                 ,vocab_size
                 ,d_model
                 ,name='decoder'):
        super(Decoder,self).__init__(name=name)
        self.nb_layers=nb_layers
        #self.FFN_units=FFN_units
        #self.nb_proj=nb_proj
        #self.dropout=dropout
        self.vocab_size=vocab_size
        self.d_model=d_model
        
        self.embedding=layers.Embedding(vocab_size,d_model)
        self.dropout=layers.Dropout(rate=dropout)
        self.pos_encoding=PositionalEncoding()
        
        self.dec_layers=[DecoderLayer(FFN_units,nb_proj,dropout) for _ in range(self.nb_layers)]
        
        
    
    

    def call(self,inputs,enc_outputs,mask_1,mask_2,training):
        
        outputs=self.embedding(inputs)
        outputs*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        outputs=self.pos_encoding(outputs)
        outputs=self.dropout(outputs,training)
        
        for i in range(self.nb_layers):
            outputs=self.dec_layers[i](outputs,
                                   enc_outputs,
                                   mask_1,
                                   mask_2,
                                   training)
        
        return outputs
    
    

class Transformer(tf.keras.Model):
    
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 nb_layers,
                 FFN_units,
                 nb_proj,
                 dropout,
                 name='transformer'):
        
        super(Transformer,self).__init__(name=name)
        
        self.encoder=Encoder(nb_layers
                             ,FFN_units
                             ,nb_proj
                             ,dropout
                             ,vocab_size_enc
                             ,d_model)
        self.decoder=Decoder(nb_layers
                             ,FFN_units
                             ,nb_proj
                             ,dropout
                             ,vocab_size_dec
                             ,d_model)
        
        self.last_linear=layers.Dense(units=vocab_size_dec)
        
    def create_padding_mask(self,seq): #seq:(batchsize,seq_length)
        mask= tf.cast(tf.math.equal(seq,0),tf.float32)
        return mask[:,tf.newaxis,tf.newaxis,:]
    
    def create_look_ahead_mask(self,seq):
        
        seq_len=tf.shape(seq)[1]
        look_ahead_mask=1-tf.linalg.band_part(tf.ones((seq_len,seq_len)),-1,0)   ## Masking the upper traiangle
        
        return look_ahead_mask
    
    def call(self, enc_inputs,dec_inputs,training):
        enc_mask=self.create_padding_mask(enc_inputs)
        dec_mask_1=tf.maximum(self.create_look_ahead_mask(dec_inputs),
                              self.create_padding_mask(dec_inputs))
        dec_mask_2=self.create_padding_mask(enc_inputs)
        
        enc_outputs=self.encoder(enc_inputs
                                 ,enc_mask
                                 ,training)
        dec_outputs=self.decoder(dec_inputs
                                 ,enc_outputs
                                 ,dec_mask_1
                                 ,dec_mask_2
                                 ,training)
        
        outputs=self.last_linear(dec_outputs)
        
        return outputs

'''
Part 5: Model Building, compilation and custome scheduling
'''
###Training
      
tf.keras.backend.clear_session()

##Hyper-Parameters

D_MODEL=128 #512
NB_LAYERS=4 #6
FFN_UNITS=512 #2048
NB_PROJ=8 #8
DROPOUT=0.1 #0.1


transformer=Transformer(vocab_size_enc=vocab_size_en,
                 vocab_size_dec=vocab_size_fr,
                 d_model=D_MODEL,
                 nb_layers=NB_LAYERS,
                 FFN_units=FFN_UNITS,
                 nb_proj=NB_PROJ,
                 dropout=DROPOUT)


loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                          reduction='none')

def loss_function(target,pred):
    
    mask=tf.math.logical_not(tf.math.equal(target,0))
    loss_=loss_object(target,pred)
    
    mask=tf.cast(mask,loss_.dtype)
    loss_*=mask
    
    return tf.reduce_mean(loss_)

train_loss=tf.keras.metrics.Mean(name="train_loss")
train_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name="training_accuracy")





class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

leaning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)


checkpoint_path= "D:/Study/Udemy/Natural Language Processing/Transformer"

ckpt=tf.train.Checkpoint(transformer=transformer,
                         optimizer=optimizer)

ckpt_manager=tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Checkpoint restored")
    
'''
Part 5: Model Building, compilation and custome scheduling

You can skip this process if you want to since I've already provinded the 
trained weigths which can be directly loaded.

'''
EPOCS=10
losses=[]
for epoch in range(EPOCS):
    print("Start of Epoch {}".format(epoch+1))
    start=time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    for(batch, (enc_inputs,targets)) in enumerate(dataset): 
        
        dec_inputs=targets[:,:-1]
        dec_outputs_real=targets[:,1:]
        
        with tf.GradientTape() as tape:
            predictions=transformer(enc_inputs,dec_inputs,True) #enc=dec=(64,20): batch_size,d_model
            loss=loss_function(dec_outputs_real,predictions)
            losses.append(loss)
        gradients=tape.gradient(loss,transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(dec_outputs_real,predictions)
        
        if batch % 50==0:
            
            print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(epoch+1,
                                                                          batch,
                                                                          train_loss.result(),
                                                                          train_accuracy.result()))
            
    ckpt_save_path=ckpt_manager.save()
    
    print("Saving chekpoint for epoch {} at {} ".format(epoch+1,ckpt_save_path))
    print("Time Taken for 1 epoch: {} seconds".format(time.time()-start))
    
          
### Save weights and models    

#transformer.save_weights('Transformer_weights.h5')

#transformer.save('Transformer_model.h5')
#transformer_arc=transformer.to_json()

#tf.keras.models.save_model(transformer,'Transformer_Model')
        

 
'''
Part 6: Model Evaluation and prediction

'''
def evaluate(inp_sentence):  ##Do we need to clean the input before eval
    inp_sentence = \
        [vocab_size_en-2] + tokenizer_en.encode(inp_sentence) + [vocab_size_en-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)
    
    output = tf.expand_dims([vocab_size_fr-2], axis=0)
    
    print(enc_input)
    print(output)
    
    for _ in range(max_length):
        
        predictions = transformer(enc_input, output, False)   ##(1, seq_length, vocab_size_fr)
        
        prediction = predictions[:, -1:, :]
        
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)
        
        if predicted_id == vocab_size_fr-1:
            return tf.squeeze(output, axis=0)
        
        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)



def translate(sentence):
    output = evaluate(sentence).numpy()
    
    predicted_sentence = tokenizer_fr.decode(
        [i for i in output if i < vocab_size_fr-2]
    )
    
    return predicted_sentence
    #print("Input: {}".format(sentence))
    #print("Predicted translation: {}".format(predicted_sentence))

'''

Part 7: Model Building using saved weights

The following steps are done when there is the saved weights and tokenizers 
are already available and you don't want to re-train the model.

'''
###Loading saved tokenizers

tokenizer_en=tfds.features.text.SubwordTextEncoder.load_from_file('tokenizer_en')
tokenizer_fr=tfds.features.text.SubwordTextEncoder.load_from_file('tokenizer_fr')


vocab_size_en=tokenizer_en.vocab_size + 2
vocab_size_fr=tokenizer_fr.vocab_size + 2

    
sentence="I wanted to buy a car but instead end up renting it"
    
inp=[vocab_size_en-2] + tokenizer_en.encode(sentence) + [vocab_size_en-1]

inp=tf.expand_dims(inp,axis=0)
inp=tf.keras.preprocessing.sequence.pad_sequences(inp,
                                                     value=0,
                                                     padding='post',
                                                     maxlen=20)
inp=tf.expand_dims(inp,axis=0)



out = tf.expand_dims([vocab_size_fr-2], axis=0)

transformer(inp,out,True)

###Loading pretrained weights#####
transformer.load_weights('Transformer_weights.h5')
 

'''
###Sanity test

translate("I wanted to buy a car but instead end up renting it")

translate("It's a wonderful day today")

'''

##################Model Deploymenet################

'''
Part 8: Deplying the model in local.

'''
import requests
from flask import Flask, request, jsonify



app = Flask(__name__)
@app.route("/api/v1/<string:input_Sentence>", methods=["POST"])

def classify_image(input_Sentence):
    '''
    in case of image file stored on local
    
    #Define the uploads folder
    upload_dir = "uploads/"
    #Load an uploaded image
    image = imread(upload_dir + img_name)
    '''

    prediction = translate(input_Sentence)

    #Return the prediction to the user
    return jsonify({"Original Sentence":input_Sentence,
                    "French translation": prediction})

#Start the Flask application
app.run(port=5000, debug=False)

##### 