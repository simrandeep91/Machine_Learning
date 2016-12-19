import numpy as np
import matplotlib.pyplot as plt
from utils import *
from plot_digits import *
from run_knn import run_knn

#SCRIPT TO RUN KNN

train_data, train_labels = load_test();
np.set_printoptions(threshold='nan')
print(train_data)
print(train_labels)
print(train_data.shape)
print(train_labels.shape)
valid_data, valid_labels = load_valid();
#test_data, test_labels = load_test();

valid1 = run_knn(9, train_data, train_labels, valid_data);

print valid_labels.shape
#print valid1
print valid1.shape;
cl1 = 0;
for i in range(0,valid1.shape[0]):
    if valid1[i,0] == valid_labels[i,0]:
        cl1 = cl1 + 1;
cl1= (cl1 * 1.0) / valid1.shape[0];

# valid3 = run_knn(3, train_data, train_labels, valid_data);
# 
# cl3 = 0;
# for i in range(0,valid3.shape[0]):
#     if valid3[i,0] == valid_labels[i,0]:
#         cl3 = cl3 + 1;
# cl3= (cl3 * 1.0) / valid3.shape[0];
# 
# valid5 = run_knn(5, train_data, train_labels, valid_data);
# 
# cl5 = 0;
# for i in range(0,valid5.shape[0]):
#     if valid5[i,0] == valid_labels[i,0]:
#         cl5 = cl5 + 1;
# cl5= (cl5 * 1.0) / valid5.shape[0];
# 
# valid7 = run_knn(7, train_data, train_labels, valid_data);
# 
# cl7 = 0;
# for i in range(0,valid7.shape[0]):
#     if valid7[i,0] == valid_labels[i,0]:
#         cl7 = cl7 + 1;
# cl7= (cl7 * 1.0) / valid7.shape[0];
# 
# valid9 = run_knn(9, train_data, train_labels, valid_data);
# 
# cl9 = 0;
# for i in range(0,valid9.shape[0]):
#     if valid9[i,0] == valid_labels[i,0]:
#         cl9 = cl9 + 1;
# cl9 = (cl9 * 1.0) / valid9.shape[0];

Y = [cl1];
X = [1]

plt.scatter(X, Y, label='Rate of Classification for Validation data')
plt.xlabel('K')
plt.ylabel('Classification Rate')
plt.title('KNN - Classification Rate vs value of K')
plt.legend();
plt.show()

# 
# test5 = run_knn(5, train_data, train_labels, test_data);
# 
# t5 = 0;
# for i in range(0,test5.shape[0]):
#     if test5[i,0] == test_labels[i,0]:
#         t5 = t5 + 1;
# t5 = (t5 * 1.0) / test5.shape[0];
# 
# test1 = run_knn(1, train_data, train_labels, test_data);
# 
# t1 = 0;
# for i in range(0,test1.shape[0]):
#     if test1[i,0] == test_labels[i,0]:
#         t1 = t1 + 1;
# t1 = (t1 * 1.0) / test1.shape[0];
# 
# test3 = run_knn(3, train_data, train_labels, test_data);
# 
# t3 = 0;
# for i in range(0,test3.shape[0]):
#     if test3[i,0] == test_labels[i,0]:
#         t3 = t3 + 1;
# t3 = (t3 * 1.0) / test3.shape[0];
# 
# test7 = run_knn(7, train_data, train_labels, test_data);
# 
# t7 = 0;
# for i in range(0,test7.shape[0]):
#     if test7[i,0] == test_labels[i,0]:
#         t7 = t7 + 1;
# t7 = (t7 * 1.0) / test7.shape[0];
# 
# test9 = run_knn(9, train_data, train_labels, test_data);
# 
# t9 = 0;
# for i in range(0,test9.shape[0]):
#     if test9[i,0] == test_labels[i,0]:
#         t9 = t9 + 1;
# t9 = (t9 * 1.0) / test9.shape[0];
# 
# Y = [t1,t3,t5,t7,t9];
# X = [1,3,5,7,9];
# 
# plt.scatter(X, Y, label='Rate of Classification for Test data')
# plt.xlabel('K')
# plt.ylabel('Classification Rate')
# plt.title('KNN - Classification Rate vs value of K')
# plt.legend();
# plt.show()