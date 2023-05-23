# Decision Tree Classifier (Pruning/Non-Pruning)
Academic decision tree program to classify data.

Train.txt is a 6x140 matrix, where the rows are attributes and columns are customers. Rows 1 – 6
correspond to attributes RISK. AGE, CRED_HIS, INCOME, RACE and HEALTH in that order. The class label
attribute is RISK. The values for each attribute have been encoded as integers.
The file test.txt is a 6x70 matrix. The interpretation for its rows and columns is identical to that for
train.txt.
The file dataDisc.txt contains a list of JSON dumped data that can be used to interpret train.txt and test.txt.

The data model is this:
[['RISK',(1,2)], ['AGE',(1,2,3)], ['CRED_HIS',(1,2)], ['INCOME',(1,2)], ['RACE',(1,2,3)], ['HEALTH',(1,2)]]

This model contains as its members six lists corresponding to the six attributes in train.txt/test.txt. Note that
these lists are in the same order as the corresponding rows in the matrixes for files train.txt and test.txt.

The tuple following each attribute name is the domain for that attribute.

The file deDomain.txt contains a nested dictionary that can be used to decode the values for the attributes.
