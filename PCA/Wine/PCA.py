import pandas as pd
import numpy as np

data = pd.read_csv("Wine.csv")

#first five rows 
data.head()

data.isnull().sum()

#seprating the target and independent variable
X = data.drop("Customer_Segment",axis=1)
y = data.Customer_Segment

# normalisation
from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler()
X = scalar.fit_transform(X)

#PCA
from sklearn.decomposition import PCA
reduce = PCA(n_components=3)
X = reduce.fit_transform(X)
variance = reduce.explained_variance_ratio_

#training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

#model building
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

# accuracy score without PCA
import pandas as pd
import numpy as np

data = pd.read_csv("Wine.csv")
#seprating the target and independent variable
X = data.drop("Customer_Segment",axis=1)
y = data.Customer_Segment

#training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

#model building
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# now you will see that when you apply PCA the accuracy is 98 approx and when you dont apply PCA 
#the accuracy is 94 approx





