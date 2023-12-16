# !pip install -q nltk git+https://github.com/salaniz/pycocoevalcap

import pickle
import numpy as np
import pandas as pd
import os
import string
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
# from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
# from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
# from tensorflow.keras.applications.xception import Xception,preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,Add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.saving import load_model
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
%matplotlib inline

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

def define_model(vocab_size,max_length,embd,glove_embedding_matrix,last_layer_units):
	unit_info=512
	inputs1=Input(shape=(last_layer_units,))
	fe1=Dropout(0.5)(inputs1)
	fe2=Dense(unit_info,activation='relu',kernel_regularizer=l2(0.01))(fe1)
	inputs2=Input(shape=(max_length,))
	if embd==0:
		se1=Embedding(vocab_size,100,mask_zero=True)(inputs2)		
	else:
		se1=Embedding(vocab_size,100,mask_zero=True,weights=[glove_embedding_matrix],trainable=False)(inputs2)
	se2=Dropout(0.5)(se1)
	se3=LSTM(unit_info)(se2)
	decoder1=Add()([fe2,se3])
	decoder2=Dense(unit_info,activation='relu',kernel_regularizer=l2(0.01))(decoder1)
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
	in_text=in_text.replace('startseq','')
	in_text=in_text.replace('endseq','')
	in_text=in_text.strip()
	return in_text


# ================================
# raw images
# ================================
all_images=os.listdir('all_images/8k/')
captions_desc=pd.read_csv('results.csv',error_bad_lines=False,sep='|')
captions_desc.columns=[str(x).strip().lower() for x in captions_desc.columns]
captions_desc=captions_desc[['image_name','comment']]
captions_desc=captions_desc[captions_desc['image_name'].isin(all_images)]
descriptions=captions_desc.groupby(by=['image_name'])['comment'].agg(list).reset_index().to_dict(orient='records')

# ================================
# split of 80:20 train:val
# ================================
train_size=int(len(descriptions)*0.80)
train_descriptions={i['image_name'].split('.')[0]:clean_description(desc=i['comment'],train_flag=True) for i in descriptions[0:train_size]}
test_descriptions={i['image_name'].split('.')[0]:clean_description(desc=i['comment'],train_flag=False) for i in descriptions[train_size:]}
vocab1=to_vocab(desc=train_descriptions)
test_vocab=to_vocab(desc=test_descriptions)
vocab1.update(test_vocab)

# ================================
# tokenization process
# ================================
tokenizer=create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index)+1
max_length=50

# ================================
# feature extraction models
# ================================

# 224 x 224
vgg_16_feature_extraction_model=VGG16()
vgg_16_feature_extraction_model=Model(inputs=vgg_16_feature_extraction_model.inputs,outputs=vgg_16_feature_extraction_model.layers[-2].output)

# resnet50_feature_extraction_model=ResNet50()
# resnet50_feature_extraction_model=Model(inputs=resnet50_feature_extraction_model.inputs,outputs=resnet50_feature_extraction_model.layers[-2].output)

# # 299 x 299
# inception_v3_feature_extraction_model=InceptionV3()
# inception_v3_feature_extraction_model=Model(inputs=inception_v3_feature_extraction_model.inputs,outputs=inception_v3_feature_extraction_model.layers[-2].output)

# xception_feature_extraction_model=Xception()
# xception_feature_extraction_model=Model(inputs=xception_feature_extraction_model.inputs,outputs=xception_feature_extraction_model.layers[-2].output)


# note:
# without super resolution input images

# ================================
# get image features
# ================================
img_features=extract_features_vgg16(feature_extraction_model=vgg_16_feature_extraction_model,directory='all_images/8k/',train_flag=True)
# img_features=extract_features_vgg16(feature_extraction_model=resnet50_feature_extraction_model,directory='all_images/8k/',train_flag=True)
# img_features=extract_features_inception_v3(feature_extraction_model=inception_v3_feature_extraction_model,directory='all_images/8k/',train_flag=True)
# img_features=extract_features_inception_v3(feature_extraction_model=xception_feature_extraction_model,directory='all_images/8k/',train_flag=True)

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
EPOCHS=600
BATCH_SIZE=16
filepath='model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

# ================================
# define the model without GLOVE
# ================================
model=define_model(vocab_size,max_length,0,None,4096) # vgg16
# model=define_model(vocab_size,max_length,0,None,2048) # resnet50
# model=define_model(vocab_size,max_length,0,None,2048) # inceptionv3
# model=define_model(vocab_size,max_length,0,None,1024) # xception

history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Train Loss vs. Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ================================
# define the model with GLOVE
# ================================
# glove_embeddings_mapping=dict()

# glove_embeddings_size=50
# with open(file='glove.6B.50d.txt',mode='r',encoding='utf-8') as inputstream:
# 	for text in inputstream:
# 		text=text.split()
# 		glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

glove_embeddings_size=100
with open(file='glove.6B.100d.txt',mode='r',encoding='utf-8') as inputstream:
    for text in inputstream:
        text=text.split()
        glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

# glove_embeddings_size=200
# with open(file='glove.6B.200d.txt',mode='r',encoding='utf-8') as inputstream:
#     for text in inputstream:
#         text=text.split()
#         glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

glove_embedding_matrix=np.zeros(shape=(vocab_size,glove_embeddings_size))
for txt,idx in tokenizer.word_index.items():
	if txt in glove_embeddings_mapping:
		glove_embedding_matrix[idx]=glove_embeddings_mapping[txt]

model=define_model(vocab_size,max_length,1,glove_embedding_matrix,4096) # vgg16
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,2048) # resnet50
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,2048) # inceptionv3
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,1024) # xception

history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Train Loss vs. Validation Loss (GloVe)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()






# note:
# with super resolution input images

# ================================
# get image features
# ================================
img_features=extract_features_vgg16(feature_extraction_model=vgg_16_feature_extraction_model,directory='ddnm_op/8k/',train_flag=True)
# img_features=extract_features_vgg16(feature_extraction_model=resnet50_feature_extraction_model,directory='ddnm_op/8k/',train_flag=True)
# img_features=extract_features_inception_v3(feature_extraction_model=inception_v3_feature_extraction_model,directory='ddnm_op/8k/',train_flag=True)
# img_features=extract_features_inception_v3(feature_extraction_model=xception_feature_extraction_model,directory='ddnm_op/8k/',train_flag=True)

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
EPOCHS=600
BATCH_SIZE=16
filepath='model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint=ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

# ================================
# define the model without GLOVE
# ================================
model=define_model(vocab_size,max_length,0,None,4096) # vgg16
# model=define_model(vocab_size,max_length,0,None,2048) # resnet50
# model=define_model(vocab_size,max_length,0,None,2048) # inceptionv3
# model=define_model(vocab_size,max_length,0,None,1024) # xception

history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Train Loss vs. Validation Loss (DDNM)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# ================================
# define the model with GLOVE
# ================================
# glove_embeddings_mapping=dict()

# glove_embeddings_size=50
# with open(file='glove.6B.50d.txt',mode='r',encoding='utf-8') as inputstream:
# 	for text in inputstream:
# 		text=text.split()
# 		glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

glove_embeddings_size=100
with open(file='glove.6B.100d.txt',mode='r',encoding='utf-8') as inputstream:
    for text in inputstream:
        text=text.split()
        glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

# glove_embeddings_size=200
# with open(file='glove.6B.200d.txt',mode='r',encoding='utf-8') as inputstream:
#     for text in inputstream:
#         text=text.split()
#         glove_embeddings_mapping[text[0]]=np.asarray(text[1:],dtype='float32')

glove_embedding_matrix=np.zeros(shape=(vocab_size,glove_embeddings_size))
for txt,idx in tokenizer.word_index.items():
	if txt in glove_embeddings_mapping:
		glove_embedding_matrix[idx]=glove_embeddings_mapping[txt]

model=define_model(vocab_size,max_length,1,glove_embedding_matrix,4096) # vgg16
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,2048) # resnet50
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,2048) # inceptionv3
# model=define_model(vocab_size,max_length,1,glove_embedding_matrix,1024) # xception

history=model.fit([X1train,X2train],ytrain,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=1,callbacks=[checkpoint],validation_data=([X1test,X2test],ytest))
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('Train Loss vs. Validation Loss (DDNM + GloVe)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



# ================================
# test on image
# ================================
model_without_glove=load_model('model1.h5')
model_with_glove=load_model('model2.h5')
model_with_ddnm=load_model('model3.h5')
model_with_ddnm_glove=load_model('model4.h5')

# ================================
# get features - use the same that you used to train the model
# ================================

all_predictions=dict()
actual_captions=dict()
without_glove_captions=dict()
with_glove_captions=dict()
with_super_resolution_captions=dict()
with_super_resolution_glove_captions=dict()

for test_case in os.listdir('unseen_images/'):
	test_case=f'unseen_images/{test_case}'
	photo_features=extract_features_vgg16(feature_extraction_model=vgg_16_feature_extraction_model,directory=test_case,train_flag=False)
	cap=captions_desc[captions_desc['image_name']==test_case]['comment'].tolist()
	wo_glv=generate_desc(model1,tokenizer,photo_features,max_length)
	w_glv=generate_desc(model2,tokenizer,photo_features,max_length)
	w_srp=generate_desc(model3,tokenizer,photo_features,max_length)
	w_srp_glv=generate_desc(model4,tokenizer,photo_features,max_length)
	actual_captions[test_case]=cap
	without_glove_captions[test_case]=wo_glv
	with_glove_captions[test_case]=w_glv
	with_super_resolution_captions[test_case]=w_srp
	with_super_resolution_glove_captions[test_case]=w_srp_glv
	all_predictions[test_case]={'captions':cap,'without glove':wo_glv,'with glove':w_glv,'with super resolution':w_srp,'with super resolution and glove':w_srp_glv}

references = {i: actual_captions[i] for i in actual_captions}
hypotheses1 = {i: [without_glove_captions[i]] for i in actual_captions}
hypotheses2 = {i: [with_glove_captions[i]] for i in actual_captions}
hypotheses3 = {i: [with_super_resolution_captions[i]] for i in actual_captions}
hypotheses4 = {i: [with_super_resolution_glove_captions[i]] for i in actual_captions}

bleu_scorer = Bleu()
cider_scorer = Cider()
meteor_scorer = Meteor()

bleu_score, _ = bleu_scorer.compute_score(references, hypotheses1)
cider_score, _ = cider_scorer.compute_score(references, hypotheses1)
meteor_score, _ = meteor_scorer.compute_score(references, hypotheses1)
print('Without GloVe:')
print(f'BLEU Score: {bleu_score}')
print(f'CIDEr Score: {cider_score}')
print(f'METEOR Score: {meteor_score}')

bleu_score, _ = bleu_scorer.compute_score(references, hypotheses2)
cider_score, _ = cider_scorer.compute_score(references, hypotheses2)
meteor_score, _ = meteor_scorer.compute_score(references, hypotheses2)
print('With GloVe:')
print(f'BLEU Score: {bleu_score}')
print(f'CIDEr Score: {cider_score}')
print(f'METEOR Score: {meteor_score}')

bleu_score, _ = bleu_scorer.compute_score(references, hypotheses3)
cider_score, _ = cider_scorer.compute_score(references, hypotheses3)
meteor_score, _ = meteor_scorer.compute_score(references, hypotheses3)
print('With Super Resolution:')
print(f'BLEU Score: {bleu_score}')
print(f'CIDEr Score: {cider_score}')
print(f'METEOR Score: {meteor_score}')

bleu_score, _ = bleu_scorer.compute_score(references, hypotheses4)
cider_score, _ = cider_scorer.compute_score(references, hypotheses4)
meteor_score, _ = meteor_scorer.compute_score(references, hypotheses4)
print('With Super Resolution and GloVe:')
print(f'BLEU Score: {bleu_score}')
print(f'CIDEr Score: {cider_score}')
print(f'METEOR Score: {meteor_score}')

