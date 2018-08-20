
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


x = np.arange(10)
x


# In[4]:


X = np.arange(15).reshape(3,5)
X


# ### 基本属性

# In[5]:


x.ndim


# In[6]:


X.ndim


# In[7]:


x.shape


# In[8]:


X.shape


# In[9]:


x.size


# In[10]:


X.size


# ### numpy.array的数据访问

# In[11]:


x


# In[12]:


x[0]


# In[13]:


x[-1]


# In[14]:


X


# In[15]:


X[0][0]


# In[16]:


X[2,2]


# In[17]:


x[0:5]


# In[18]:


x[:5]


# In[19]:


x[5:]


# In[20]:


x[::2]


# In[21]:


x[::-1]


# In[22]:


X


# In[23]:


X[:2,:3]


# In[24]:


X[:2][:3]


# In[25]:


X[:2]


# In[26]:


X[:2,::2]


# In[27]:


X[0]


# In[28]:


X[0,:]


# In[29]:


X[0,:].ndim


# In[30]:


X[:,0]


# In[31]:


X[:,0].ndim


# In[32]:


subX = X[:2,:3]
subX


# In[33]:


subX[0,0] = 100
subX


# In[34]:


X


# In[35]:


X[0,0] = 0
X


# In[36]:


subX


# In[37]:


subX = X[:2,:3].copy()
subX


# In[38]:


subX[0,0] = 100
subX


# In[39]:


X


# ### Reshape

# In[40]:


x.shape


# In[41]:


x.ndim


# In[42]:


x.reshape(2,5)


# In[43]:


x


# In[44]:


A = x.reshape(2,5)
A


# In[45]:


x


# In[46]:


B = x.reshape(1,10)
B


# In[47]:


B.shape


# In[48]:


B.ndim


# In[49]:


x.shape


# In[50]:


x.reshape(10,-1)


# In[51]:


x.reshape(-1,10)


# In[52]:


x.reshape(2,-1)


# ### 合并操作

# In[53]:


x = np.array([1,2,3])
y = np.array([3,2,1])


# In[54]:


x


# In[55]:


y


# In[56]:


np.concatenate([x,y])


# In[57]:


z = np.array([666,666,666])


# In[58]:


np.concatenate([x , y ,z])


# In[59]:


A = np.array([[1, 2 ,3],
             [4, 5, 6]])


# In[60]:


np.concatenate([A, A])


# In[61]:


np.concatenate([A , A],axis=1)


# In[62]:


np.concatenate([A, z])


# In[63]:


np.concatenate([A, z.reshape(1,-1)])


# In[64]:


A


# In[65]:


A2 = np.concatenate([A, z.reshape(1,-1)])


# In[66]:


A2


# In[67]:


np.vstack([A, z])


# In[68]:


B = np.full((2,2), 100)


# In[69]:


B


# In[70]:


np.hstack([A, B])


# ### 分割

# In[71]:


x = np.arange(10)
x


# In[72]:


x1, x2, x3 = np.split(x,[3, 7])


# In[73]:


x


# In[74]:


x1


# In[75]:


x2


# In[76]:


x3


# In[77]:


x1, x2 = np.split(x, [5])


# In[78]:


x1


# In[79]:


x2


# In[80]:


A = np.arange(16).reshape((4,4))
A


# In[81]:


A1, A2 = np.split(A,[2])


# In[82]:


A1


# In[83]:


A2


# In[84]:


A1, A2 = np.split(A,[2], axis = 1)


# In[85]:


A1


# In[86]:


A2


# In[87]:


upper, lower = np.vsplit(A, [2])


# In[88]:


upper


# In[89]:


lower


# In[90]:


left, right = np.hsplit(A,[2])


# In[91]:


left


# In[92]:


right


# In[93]:


data = np.arange(16).reshape((4,4))
data


# In[94]:


X, y = np.hsplit(data,[-1])


# In[95]:


X


# In[96]:


y


# In[97]:


y[:,0]


# ## numpy.array中的运算

# #### 给定一个向量，让向量中每一个数乘以2
# a = （0，1，2）
# a*2 = （0，2，4)

# In[98]:


n = 10
L = [i for i in range(n)]


# In[99]:


2*L


# In[100]:


A = []
for e in L:
    A.append(2*e)
A


# In[101]:


n = 1000000
L = [i for i in range(n)]


# In[102]:


get_ipython().run_cell_magic('time', '', 'A = []\nfor e in L:\n    A.append(2*e)')


# In[103]:


get_ipython().run_cell_magic('time', '', 'A = [2*e for e in L]')


# In[104]:


L = np.arange(n)


# In[105]:


get_ipython().run_cell_magic('time', '', 'A = np.array(2*e for e in L)')


# In[106]:


get_ipython().run_cell_magic('time', '', 'A = 2 * L')


# In[107]:


A


# In[108]:


n = 10
L = np.arange(n)
2 * L


# ## Universal Function

# In[109]:


X = np.arange(1,16).reshape((3, 5))
X


# In[110]:


X + 1


# In[111]:


X - 1


# In[112]:


X * 2 


# In[113]:


X / 2


# In[114]:


X // 2


# In[115]:


X ** 2


# In[116]:


X % 2


# In[117]:


1 / X


# In[118]:


np.abs(X)


# In[119]:


np.sin(X)


# In[120]:


np.exp(X)


# In[121]:


np.power(3, X)


# In[122]:


3 ** X


# In[123]:


np.log(X)


# In[124]:


np.log2(X)


# In[125]:


np.log10(X)


# ## 矩阵运算

# In[126]:


A = np.arange(4).reshape(2, 2)
A


# In[127]:


B = np.full((2,2), 10)
B


# In[128]:


A + B


# In[129]:


A - B


# In[130]:


A * B


# In[131]:


A / B


# In[132]:


A.dot(B)


# In[133]:


A 


# In[134]:


A.T


# In[135]:


c = np.full((3, 3), 666)


# ## 向量和矩阵的运算

# In[136]:


v = np.array([1, 2])


# In[137]:


A


# In[138]:


v + A


# In[139]:


np.vstack([v] * A.shape[0])


# In[140]:


np.vstack([v] * A.shape[0]) + A


# In[141]:


np.tile(v, (2, 1))


# In[142]:


v


# In[143]:


A


# In[144]:


v * A


# In[145]:


v.dot(A)


# In[146]:


A.dot(v)


# ## 矩阵的逆

# In[147]:


A


# In[148]:


np.linalg.inv(A)


# In[149]:


invA= np.linalg.inv(A)


# In[150]:


A.dot(invA)


# In[151]:


X


# In[152]:


np.linalg.inv(X)


# In[153]:


pinvX = np.linalg.pinv(X)


# In[154]:


pinvX


# In[155]:


pinvX.shape


# In[156]:


X.dot(pinvX)


# ## 聚合操作

# In[157]:


L = np.random.random(100)


# In[158]:


L


# In[159]:


sum(L)


# In[160]:


np.sum(L)


# In[161]:


big_array = np.random.rand(1000000)
get_ipython().run_line_magic('timeit', 'sum(big_array)')
get_ipython().run_line_magic('timeit', 'np.sum(big_array)')


# In[162]:


np.min(big_array)


# In[163]:


np.max(big_array)


# In[164]:


big_array.min()


# In[165]:


big_array.sum()


# In[166]:


X = np.arange(16).reshape(4,-1)
X


# In[167]:


np.sum(X)


# In[168]:


np.sum(X ,axis = 1)


# In[169]:


np.sum(X ,axis = 0)


# In[170]:


np.prod(X)


# In[171]:


np.prod(X + 1)


# In[172]:


np.mean(X)


# In[173]:


np.median(X)


# In[174]:


v = np.array([1, 1, 2, 2, 10])
np.mean(v)


# In[175]:


np.median(v)


# In[176]:


np.percentile(big_array, q = 50)


# In[177]:


np.median(big_array)


# In[178]:


np.percentile(big_array, q = 100)


# In[179]:


np.max(big_array)


# In[180]:


for percent in [0, 25, 50, 75, 100]:
    print(np.percentile(big_array, q = percent))


# In[181]:


np.var(big_array)


# In[182]:


np.std(big_array)


# In[183]:


x = np.random.normal(0, 1, size = 1000000)


# In[184]:


np.mean(x)


# In[185]:


np.std(x)


# ## 索引

# In[186]:


np.min(x)


# In[187]:


np.argmin(x)


# In[188]:


x[720143]


# In[189]:


np.argmax(x)


# In[190]:


x[693938]


# In[191]:


np.max(x)


# ## 排序和使用索引

# In[192]:


x = np.arange(16)
x


# In[193]:


np.random.shuffle(x)
x


# In[194]:


np.sort(x)


# In[195]:


x


# In[196]:


x.sort()


# In[197]:


x


# In[198]:


X = np.random.randint(10, size = (4, 4))
X


# In[199]:


np.sort(X)


# In[200]:


np.sort(X, axis = 1)


# In[201]:


np.sort(X, axis = 0)


# In[202]:


x = np.arange(16)
x


# In[203]:


np.random.shuffle(x)


# In[204]:


x


# In[205]:


np.argsort(x)


# In[206]:


np.partition(x, 3)


# In[207]:


np.argpartition(x, 3)


# In[208]:


X


# In[209]:


np.argsort(X, axis = 1)


# In[210]:


np.argpartition(X , 2, axis = 1)


# ## Fancy Indexing

# In[211]:


x = np.arange(16)
x


# In[212]:


x[3]


# In[213]:


x[3:9]


# In[214]:


x[3:9:2]


# In[215]:


[x[3],x[5],x[8]]


# In[216]:


ind = [3, 5, 8]


# In[217]:


x[ind]


# In[218]:


ind = np.array([[0, 2],
              [1, 3]])
x[ind]


# In[219]:


X = x.reshape(4, -1)
X


# In[220]:


row = np.array([0, 1 ,2])
col = np.array([1, 2 ,3])
X[row, col]


# In[221]:


X[0, col]


# In[222]:


X[:2, col]


# In[223]:


col = [True, False, True, True]


# In[224]:


X[1:3, col]


# ## numpy.array的比较

# In[225]:


x


# In[226]:


x < 3


# In[227]:


x <= 3


# In[228]:


x == 3


# In[229]:


x != 3


# In[230]:


2 * x == 24- 4 * x


# In[231]:


X


# In[232]:


X < 6


# In[233]:


x


# In[234]:


np.sum(x <= 3)


# In[235]:


np.count_nonzero(x <= 3)


# In[236]:


np.any(x == 0)


# In[237]:


np.any(x < 1)


# In[238]:


np.all(x >= 0)


# In[239]:


np.all(x > 0)


# In[240]:


X


# In[241]:


np.sum(X % 2 == 0)


# In[242]:


np.sum(X % 2 == 0,axis = 1)


# In[243]:


np.sum(X % 2 == 0,axis = 0)


# In[244]:


np.all(X > 0,axis = 1)


# In[245]:


x


# In[246]:


np.sum((x > 3) & (x < 10))


# In[247]:


np.sum((x > 3) && (x < 10))


# In[248]:


np.sum((x % 2 == 0) | (x > 10))


# In[249]:


np.sum(~(x == 0))


# In[250]:


x[x < 5]


# In[251]:


x[x % 2 == 0]


# In[252]:


X


# In[253]:


X[X[:,3] % 3 == 0,:]

