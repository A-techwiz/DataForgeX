# DataForgeX
Telco customer churn: IBM dataset

Background:
With the enormous increase in the number of customers using telephone services, the marketing division of
telco company is aiming to attract more new customers and avoid contract termination from existing customers. In this case,
churn rate is the termination of the contract. The telco company intends to expand its clientele and its growth rate. 
The number of new customers must exceed its churn rate i.e. number of customers existing.


Problem Statement:
A high churn rate will adversely affect a company’s profits and impede growth. Hence a churn prediction would be 
able to provide clarity to the telco company on how well it is retaining its existing customers and understand what 
are the underlying reasons that are causing existing customers to terminate their contract i.e. a higher churn rate.


Introduction:

This project aims to analyze the telecommunication customer churn dataset to provide 
actionable insights to retain customers and increase customers’ lifetime values. For this purpose, a few quesitons
were formulated after assesing the dataset that I have attempted to investigate.

Research Questions:

Following are the research questions that I have attempted to answer.

Q1- How many customers have subscribed to what type of internet service?
Q2- What are the internet features?
Q3- What is the ratio of male and female subscribers?
Q4- What is the distribution of tenure and contract?
Q5- What are the churn rates according to the different variables in the dataset and so on. 


Dataset Information:

The dataset of telco communications company consists of the following information.
The dataset provides demographic information about customers including gender, age, marital status. It provides company specific
informaiton like Customer account information including the number of months staying with the company, 
paperless billing, payment method, monthly charges, and total charges. Customer usage behavior, such as streaming TV, streaming movie,
and other internet related services information. In addition to the above, it also sheds light upon the Services that 
the customer signed up for, such as phone service, multiples, internet service, online security, online backup, 
device protection, and tech support. And finally, Customer churn where the customer left within the last month or not. 
The dataset comprised on 7034 instances and 33 features. 

Methodology:

The churn analysis is important for the telco company to understand why the customer has stopped using its product or service. 
Unless the company understands what is the total loss of revenue caused by customer’s cancellations, which customers are canceling, 
and why they are canceling, it is hard for the telco company to improve its product and service. Hence, the research is imperative for a
nuanced understanding of the churn rate analysis and prediction. Since churn rate analysis is a typical classification problem within
the domain of supervised learning, I will be initially using Logistic Regression. To further check the accuracy of my model,
I have also implemented other alogorithms such as KNN, Naive Bayes, Random Forest, and Support Vector Machineto 
analyze customer’s churn behavior.

Instructions:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

These are the important libraries for data visualisation.

I removed noise from the data. 
Then I removed the null values from the data.
For a predictive analysis, I had to remove some columns that were not required. Examples can be seen in the Code file.
I then explored the target variable that is my Churn Value. 
After visualisation, I came to the conclusion that the number of customers who had left were less compared to
the ones who did not unsubscribe to the telecommunication service. 

Looking at the Internet service distribution, I could observe that Fiber Optic had the largest subscribers, 
compared to DSL, while sfew customers did not subscribe to the internet service. 
Looking at the contract type of customers, there were three. Month to month, Two year, and One year. 
Most of the customers had subscribed a month to month service, while some had a two year contract followed by one year.
Similalry, the payment method was also distributed into four categories. Electronic check, Mailed check, Bank transfer, and credit card. 
Most of the subscribers had preffered to pay via electronic check, while others paid through mailed check. 
The latter two options had fewer sbscribers. 
An interesting observation to note here is that customers who paid electronic check were more likely to churn while this
was also the most preferred method of customers. 

For a feature selection, I organised the categorical columns. For that purpose, I used the following code:

catcol=df.select_dtypes(include=['object']).columns.tolist()
print(catcol)
len(catcol)


This function organised the columns according to object datatype, and clustered all the categorical columns.

Afterwards, I coded the categorical columns for further processing, and to build a machine learning model.
For coding, I used one hot encoding. There was some confusion about multicolinearity and the curse of dimensionality.

Finally, I began the process for model building. I will mention the necessary libraries below.

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)

Then I checked the shape of the training and testing data. I distributed the dataset into 80:20 ratio for testing and training.
According to the split, the training data consisted of 80% of the total values of 7042 instances. 
In a similar way, the testing data comprised of 20% of the instances. 
The number of counts of the dataset can be seen in the coding file. 

Later, I applied the logistic regression to the data. For this purpose, I downloaded the following libraries.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

later on I build my model of logistic regression.
model=LogisticRegression (max_iter=700)

The max_iteration hyperparameter in the above code specifies the maximum number of iterations allowed 
for the solver to converge to the optimal solution.If the solver does not converge 
within the specified number of iterations it will stop and return the best 
solution found so far, which may not necessarily be the optimal solution.

Afterwards, I checked the accuracy of my model on the test and training data.
Below is the code used for both. 

# Accuracy score on training data
train_pred = model.predict(X_train)
acc_train = accuracy_score(train_pred, Y_train)
print("Accuracy score on trianing data:",acc_train)

Accuracy score on trianing data: 0.8105805077223505


# Accuracy score on test data
test_pred= model.predict(X_test)
acc_test= accuracy_score(test_pred,Y_test )
print('Accuracy score on test data:', acc_test)

Accuracy score on test data: 0.815471965933286


Afterwards, I checked the recall score and precision. 
Recall is known as sensitivity or true positive rate. It is the fraction of relevant instances 
that are correctly identified by a system. Recall measures the system's ability to correctly identify 
all positive instances in a dataset.
Precision is also known as positive predictive value or the fraction of retrieved instances that are actually relevant. 
In other words, precision measures the system's ability to only return relevant results in a dataset. 

To check these two measure, the nexessary libraries imported are as follows

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import metrics

Finally, I built a confusion matrix for the visualisation of the data. 
For this, I used the following code. 

confusion_matrix = metrics.confusion_matrix(Y_test, test_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = ['Negative', 'Positive'])
cm_display.plot()
plt.title('Confusion Matrix: LogisticRegression')
plt.show()

For the other algorithms, the following results were achieved.
Accuracy on Log Regression: 0.815471965933286

Accuracy of SVM: 0.8062455642299503

Accuracy on Random Forest: 0.7899219304471257
 
Accuracy on Naive Bayes is: 0.6997870830376153

Accuracy on KNN is: 0.789921930447125

