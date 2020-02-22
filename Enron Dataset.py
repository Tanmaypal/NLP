#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np


# In[ ]:





# In[84]:


df=pd.read_csv('D:\\NLP folder\\emails.csv')


# In[85]:


df.head()


# In[86]:


email_subset=df.sample(frac=0.002, random_state=1)

#print(email_subset.shape)
#print(email_subset.head()
print(email_subset.shape)
print(email_subset.head())


# In[87]:


import nltk
nltk.download('punkt')


# In[88]:


from nltk.tokenize import word_tokenize


# In[89]:


from sklearn.feature_extraction.text import CountVectorizer


# In[90]:


cv=CountVectorizer(max_df=0.95,min_df=0.1,tokenizer=word_tokenize,stop_words='english')


# In[91]:


def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    #print("lines",lines)
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            print("messages",message)
            email['body'] = message
            print("email",email)
        else:
            pairs = line.split(':')
            print("pairs",pairs)
            key = pairs[0].lower()
            print("key",key)
            val = pairs[1].strip()
            print("val",val)
            if key in keys_to_extract:
                email[key] = val
    return email


# In[92]:


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }


# In[93]:


def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


# In[94]:


email_df = pd.DataFrame(parse_into_emails(email_subset.message))
#print(email_df.head())


# In[95]:


email_df.head()


# In[96]:


import re
import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
Lemmitizer=WordNetLemmatizer()
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.corpus import wordnet
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[97]:


#punkt is used for tokenising sentences and averaged_perceptron_tagger is used for tagging words with 
#their parts of speech (POS)


# In[98]:


list1=[]
for i in range(0,len(email_df)):
  review=re.sub('[^a-zA-Z]',' ',email_df['body'][i])
  review=review.lower()
  review=word_tokenize(review)
  review = [w for w in review if not w in stop_words] 
   #review=review.split()
  pos_tag=nltk.pos_tag(review)
  list1.append(pos_tag)


# In[99]:


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


# In[100]:


lemma=[]

for word in list1:
  #print(word)
  l=[]
  for p in word:
    #print(p)
    m=Lemmitizer.lemmatize(p[0], get_wordnet_pos(p[1]))
    #print(p, m)
    l.append(m)

  l = " ".join(l)
  lemma.append(l)


# In[101]:


lemma


# In[102]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
f=tfidf.fit_transform(lemma)
features=pd.DataFrame(f.toarray(),columns=tfidf.get_feature_names())


# In[103]:


features.shape


# In[104]:


features.head()


# In[105]:


from sklearn.decomposition import  TruncatedSVD


# In[106]:


svd=TruncatedSVD(n_components=4,n_iter=100)


# In[107]:


svd.fit_transform(f)


# In[108]:


row1=svd.components_[0]
len(row1)


# In[109]:


row=svd.components_
row


# In[110]:


terms=tfidf.get_feature_names()
len(terms)


# In[111]:


for i,comp in enumerate(svd.components_):
    component_terms=zip(terms,comp)
    sortedTerms=sorted(component_terms,key=lambda x:x[1],reverse=True)
    sortedterms=sortedTerms[:10]
    print("\nconcept",i,":")
    for terms in sortedTerms:
        print(terms)


# # # NMF

# In[112]:


model=NMF(n_components=4,random_state=46,alpha=0.1, l1_ratio=0.5, init="nndsvd")


# In[113]:


H=model.fit_transform(f)


# In[114]:


W=model.components_


# In[115]:


def display_topics(H,W,feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])


# In[116]:


display_topics(W,H,tfidf.get_feature_names(),email_df['body'], 4,3)


# In[ ]:




