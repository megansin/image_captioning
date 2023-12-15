import pickle
import numpy as np
import pandas as pd
import os
import string
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import load_model
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline

def plot_history(history):
	plt.plot(history.history['loss'],label='Train Loss')
	plt.plot(history.history['val_loss'],label='Val Loss')
	plt.legend()
	plt.show()

def clean_description(desc,train_flag):
	all_captions=[]
	for caption in desc:
		caption=[ch for ch in caption if ch not in string.punctuation]
		caption=''.join(caption)
		caption=caption.split(' ')
		caption=[word.lower().strip() for word in caption if len(word)>1 and word.isalpha()]
		caption=' '.join(caption)
		if train_flag:
			caption='startseq '+caption+' endseq'
		all_captions.append(caption)
	return all_captions

def to_vocab(desc={}):
	unique_words=set()
	for i in desc.values():
		for v in i:
			unique_words.update(v.split())
	return unique_words

def extract_features_vgg16(feature_extraction_model,directory,train_flag):
	features=dict()
	if train_flag==True:
		for name in os.listdir(directory):
			filename=directory+'/'+name
			image=load_img(filename,target_size=(224,224))
			image=img_to_array(image)
			image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
			image=preprocess_input(image)
			feature=feature_extraction_model.predict(image,verbose=0)
			image_id=name.split('.')[0]
			features[image_id]=feature
	else:
		image=load_img(directory,target_size=(224,224))
		image=img_to_array(image)
		image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
		image=preprocess_input(image)
		features=feature_extraction_model.predict(image,verbose=0)
	return features

def extract_features_inception_v3(feature_extraction_model,directory,train_flag):
	features=dict()
	if train_flag == True:
		for name in os.listdir(directory):
			filename=directory+'/'+name
			image=load_img(filename,target_size=(299,299))
			image=img_to_array(image)
			image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
			image=preprocess_input(image)
			feature=feature_extraction_model.predict(image,verbose=0)
			image_id=name.split('.')[0]
			features[image_id]=feature
	else:
		image=load_img(directory,target_size=(299,299))
		image=img_to_array(image)
		image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
		image=preprocess_input(image)
		features=feature_extraction_model.predict(image,verbose=0)
	return features

def extract_features_resnet50(feature_extraction_model,directory,train_flag):
	features=dict()
	if train_flag == True:
		for name in os.listdir(directory):
			filename=directory+'/'+name
			image=load_img(filename,target_size=(224,224))
			image=img_to_array(image)
			image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
			image=preprocess_input(image)
			feature=feature_extraction_model.predict(image,verbose=0)
			image_id=name.split('.')[0]
			features[image_id]=feature
	else:
		image=load_img(directory,target_size=(224,224))
		image=img_to_array(image)
		image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
		image=preprocess_input(image)
		features=feature_extraction_model.predict(image,verbose=0)
	return features

def extract_features_xception(feature_extraction_model,directory,train_flag):
	features=dict()
	if train_flag == True:
		for name in os.listdir(directory):
			filename=directory+'/'+name
			image=load_img(filename,target_size=(299,299))
			image=img_to_array(image)
			image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
			image=preprocess_input(image)
			feature=feature_extraction_model.predict(image,verbose=0)
			image_id=name.split('.')[0]
			features[image_id]=feature
	else:
		image=load_img(directory,target_size=(299,299))
		image=img_to_array(image)
		image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
		image=preprocess_input(image)
		features=feature_extraction_model.predict(image,verbose=0)
	return features

def to_lines(descriptions):
	all_desc=list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

def create_tokenizer(descriptions):
	lines=to_lines(descriptions)
	tokenizer=Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def get_max_length(descriptions):
	lines=to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def create_sequences(tokenizer,max_length,descriptions,photos,vocab_size):
	X1,X2,y=list(),list(),list()
	for key,desc_list in descriptions.items():
		for desc in desc_list:
			seq=tokenizer.texts_to_sequences([desc])[0]
			for i in range(1,len(seq)):
				in_seq,out_seq=seq[:i],seq[i]
				in_seq=pad_sequences([in_seq],maxlen=max_length)[0]
				out_seq=to_categorical([out_seq],num_classes=vocab_size)[0]
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return np.array(X1),np.array(X2),np.array(y)

def define_model(vocab_size,max_length,glove_embedding_matrix):
	inputs1=Input(shape=(4096,))
	fe1=Dropout(0.5)(inputs1)
	fe2=Dense(256,activation='relu')(fe1)
	inputs2=Input(shape=(max_length,))
	if glove_embedding_matrix:
		se1=Embedding(vocab_size,256,mask_zero=True,weights=[glove_embedding_matrix],trainable=False)(inputs2)
	else:
		se1=Embedding(vocab_size,256,mask_zero=True)(inputs2)
	se2=Dropout(0.5)(se1)
	se3=LSTM(256)(se2)
	decoder1=Add()([fe2,se3])
	decoder2=Dense(256,activation='relu')(decoder1)
	outputs=Dense(vocab_size,activation='softmax')(decoder2)
	model=Model(inputs=[inputs1,inputs2],outputs=outputs)
	model.compile(loss='categorical_crossentropy',optimizer='adam')
	plot_model(model,to_file='model.png',show_shapes=True)
	return model

def word_for_id(integer,tokenizer):
	for word,index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model,tokenizer,photo,max_length):
	in_text='startseq'
	for i in range(max_length):
		sequence=tokenizer.texts_to_sequences([in_text])[0]
		sequence=pad_sequences([sequence],maxlen=max_length)
		yhat=model.predict([photo,sequence],verbose=0)
		yhat=np.argmax(yhat)
		word=word_for_id(yhat,tokenizer)
		if word is None:
			break
		in_text += ' '+word
		if word == 'endseq':
			break
	return in_text


all_images=os.listdir('all_images/2k/')
captions_desc=pd.read_csv('results.csv',error_bad_lines=False,sep='|')
captions_desc.columns=[str(x).strip().lower() for x in captions_desc.columns]
captions_desc=captions_desc[['image_name','comment']]
captions_desc=captions_desc[captions_desc['image_name'].isin(all_images)]

descriptions=captions_desc.groupby(by=['image_name'])['comment'].agg(list).reset_index().to_dict(orient='records')
train_size=int(len(descriptions)*0.80)
train_descriptions={i['image_name'].split('.')[0]:clean_description(desc=i['comment'],train_flag=True) for i in descriptions[0:train_size]}
test_descriptions={i['image_name'].split('.')[0]:clean_description(desc=i['comment'],train_flag=False) for i in descriptions[train_size:]}
vocab1=to_vocab(desc=train_descriptions)
test_vocab=to_vocab(desc=test_descriptions)
vocab1.update(test_vocab)

vgg_16_feature_extraction_model=VGG16()
vgg_16_feature_extraction_model=Model(inputs=vgg_16_feature_extraction_model.inputs,outputs=vgg_16_feature_extraction_model.layers[-2].output)

inception_v3_feature_extraction_model=InceptionV3()
inception_v3_feature_extraction_model=Model(inputs=inception_v3_feature_extraction_model.inputs,outputs=inception_v3_feature_extraction_model.layers[-2].output)

resnet50_feature_extraction_model=ResNet50()
resnet50_feature_extraction_model=Model(inputs=resnet50_feature_extraction_model.inputs,outputs=resnet50_feature_extraction_model.layers[-2].output)

xception_feature_extraction_model=Xception()
xception_feature_extraction_model=Model(inputs=xception_feature_extraction_model.inputs,outputs=xception_feature_extraction_model.layers[-2].output)

# ================================
# feature extraction models
# ================================
img_features=extract_features_vgg16(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/',train_flag=True)
# img_features=extract_features_inception_v3(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/',train_flag=True)
# img_features=extract_features_resnet50(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/',train_flag=True)
# img_features=extract_features_xception(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/',train_flag=True)

tokenizer=create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index)+1
max_length=get_max_length(train_descriptions)

# ================================
# training dataset
# ================================
X1train,X2train,ytrain=create_sequences(tokenizer,max_length,train_descriptions,img_features,vocab_size)

# ================================
# validation dataset
# ================================
X1test,X2test,ytest=create_sequences(tokenizer,max_length,test_descriptions,img_features,vocab_size)

# ================================
# hyper-parameters
# ================================
EPOCHS=20
BATCH_SIZE=16
filepath='model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

# ================================
# define the model without GLOVE
# ================================
model=define_model(vocab_size,max_length,None)
history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plot_history(history)

glove_embeddings_mapping=dict()

glove_embeddings_size=50
with open(file='glove.6B.50d.txt',mode='r',encoding='utf-8') as inputstream:
	for text in inputstream:
		text=text.split()
		glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

# glove_embeddings_size=100
# with open(file='glove.6B.100d.txt',mode='r',encoding='utf-8') as inputstream:
#     for text in inputstream:
#         text=text.split()
#         glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

# glove_embeddings_size=200
# with open(file='glove.6B.200d.txt',mode='r',encoding='utf-8') as inputstream:
#     for text in inputstream:
#         text=text.split()
#         glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

# glove_embeddings_size=300
# with open(file='glove.6B.300d.txt',mode='r',encoding='utf-8') as inputstream:
#     for text in inputstream:
#         text=text.split()
#         glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

glove_embedding_matrix=np.zeros(shape=(vocab_size,glove_embeddings_size))
for txt,idx in tokenizer.word_index.items():
	if txt in glove_embeddings_mapping:
		glove_embedding_matrix[idx]=glove_embeddings_mapping[txt]

# ================================
# define the model with GLOVE
# ================================
model=define_model(vocab_size,max_length,glove_embedding_matrix)
history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plot_history(history)


# ================================
# test on image
# ================================
# pass image name without ".jpg"
test_descriptions['imag1']

# pass model name that you want to load
model=load_model('model1.h5')

# ================================
# get features - use the same that you used to train the model
# ================================
photo_features=extract_features_vgg16(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/278496691.jpg',train_flag=False)
# photo_features=extract_features_inception_v3(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/278496691.jpg',train_flag=False)
# photo_features=extract_features_resnet50(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/278496691.jpg',train_flag=False)
# photo_features=extract_features_xception(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/2k/278496691.jpg',train_flag=False)

# ================================
# get caption
# ================================
description=generate_desc(model,tokenizer,photo_features,max_length)
print(description)

