
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.Series([0.25,0.5,0.75,1.0],index=["a","b","c","d"])
data


# In[3]:


data["d"]


# In[4]:


area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({"area":area,"pop":pop})
data


# In[5]:


data["area"]


# In[6]:


data.area


# In[7]:


data.area is data["area"]


# In[8]:


data.values


# In[9]:


data.T


# In[10]:


data["density"] = data["pop"] / data["area"]
data


# In[11]:


data.density is data["density"]


# In[12]:


data.T


# In[13]:


data.values


# In[14]:


data.iloc[:3,:2]


# In[16]:


import numpy as np

vals1 = np.array([1,None,3,4])
vals1


# In[18]:


for dtype in ["object","int"]:
    print("dtype =",dtype)
    get_ipython().run_line_magic('timeit', 'np.arrange(1E6,dtype=dtype).sum()')
    print()


# In[19]:


pd.Series([1,np.nan,2,None])


# In[20]:


x = pd.Series(range(2),dtype=int)
x


# In[21]:


x[0] = None
x


# In[22]:


data = pd.Series([1,np.nan,"hello",None])
data.isnull()


# In[23]:


data[data.notnull()]


# In[24]:


data.dropna()


# In[25]:


df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
df


# In[26]:


df.dropna()


# In[27]:


df.dropna(axis = "columns")


# In[28]:


df[3] = np.nan


# In[29]:


df.dropna(axis = "columns",how="all")


# In[30]:


df.dropna(axis = "rows",thresh=3)


# In[31]:


data = pd.Series([1,np.nan,2,None,3],index = list("abcde"))


# In[32]:


data


# In[33]:


data.fillna(0)


# In[34]:


data.fillna(method="ffill")


# In[35]:


data.fillna(method="bfill")


# In[36]:


df


# In[44]:


df.fillna(method="bfill",axis = "columns")


# In[45]:


index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop


# In[46]:


pop[("California",2010):("Texas",2000)]


# In[48]:


pop[[i for i in pop.index if i[1] == 2010]]


# In[49]:


index = pd.MultiIndex.from_tuples(index)
index


# In[50]:


pop = pop.reindex(index)
pop


# In[51]:


pop[:,2010]


# In[52]:


df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
df


# In[53]:


data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
pd.Series(data)


# In[54]:


pd.MultiIndex.from_arrays([["a","a","b","b"],[1,2,1,2]])


# In[55]:


pd.MultiIndex.from_tuples([("a,",1),("a",2),("b",1),("b",2)])


# In[56]:


pop.index.names = ["state","year"]
pop


# In[63]:


def make_df(cols,ind):
    data = {c:[str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data,ind)

data = make_df("ABC",range(3))
data["A"]


# In[64]:


x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
np.concatenate([x, y, z])


# In[65]:


x = [[1, 2],
     [3, 4]]
np.concatenate([x, x], axis=1)

