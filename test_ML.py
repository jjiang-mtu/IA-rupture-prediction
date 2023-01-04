import pandas as pd
import numpy as np 
import time
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import eli5   # python permutation importance tools
from eli5.sklearn import PermutationImportance

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from bartpy.sklearnmodel import SklearnModel

# A function to ignore warnings!
import warnings
warnings.simplefilter('ignore')  

def load_csv_data(filename):     # Load Data
    
    # Load Data.
    df=pd.read_csv(filename)
    df.head()
    col_names = df.columns.tolist() # Get the name of columns
    
    # Remove the space in column names
    for index,value in enumerate(col_names):
        col_names[index]= value.replace("_"," ")
        
    #col_names
    df.columns=col_names
    
    df.get('ruptureStatus')
    df['ruptureStatus'].replace(['R','U'],[1,0],inplace=True)

    missing=df[df['ruptureStatus'].isnull()]
    fill=df[df['ruptureStatus'].notnull()]
    #fill.columns = (fill.columns.str.strip().str.upper().str.replace(' ', '_'))
    
    y=fill['ruptureStatus']
    x=fill.drop(['Case','ruptureStatus'],axis=1)    
    x=pd.get_dummies(x)
    x = x.fillna(0)
    
    return x, y 
     
#------------------------------------------------------------------------------  
#--- Logistic Regression 
#------------------------------------------------------------------------------   
def test_LR(num_iters, features, labels):    

    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters)     

    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()                
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels)     
        lr=LogisticRegression(verbose=0)        
        PARAMETERS = {
             'solver': ['liblinear','lbfgs'],
             'C':[0.5,1,1.5,2,2.5,5],
             'penalty':['l1','l2'],
             'class_weight':['balanced'],
             #'multi_class': ['ovr', 'multinomial'],
             'max_iter':[100,200,300,500],
            }       
        grid = GridSearchCV(lr, param_grid=PARAMETERS, scoring='neg_log_loss', cv=10, verbose=0)#neg_log_loss,roc_auc
        grid.fit(x_train,y_train)
        y_pred=grid.predict(x_test)
            
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(grid, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)
                    
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0)) 
    
    print("-------------------------")               
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("LR_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in LR_Res.csv")
    print("****************************************************")
    
#------------------------------------------------------------------------------  
#--- Support Vector Machine  
#------------------------------------------------------------------------------  
def test_SVM(num_iters, features, labels):   

    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters)     

    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()        
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels) # 
        svc=SVC(probability=True)
        param_grid = {
             'kernel': ['linear'],
             'C':[ 0.1, 0.5, 1],
             'gamma': [0.1,0.5,1,2],
             'class_weight':['balanced'],
            # 'random_state':[800]
            }
        grid = GridSearchCV(svc, param_grid=param_grid, scoring='roc_auc', cv=5, verbose=0)#neg_log_loss,roc_auc # cv=3, acc=0.7646
        grid.fit(x_train,y_train)
        y_pred=grid.predict(x_test) 
            
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(grid, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) # 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)        
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0)) 

    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")      
    with open("SVM_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in SVM_Res.csv")
    print("****************************************************") 
    
#------------------------------------------------------------------------------  
#--- Random Forest  
#------------------------------------------------------------------------------  
def test_RF(num_iters, features, labels):    

    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters)     

    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()        
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels)         
        rf=RandomForestClassifier()
        param_grid = {
             "n_estimators":[50,100,200,301,401,500,800],#****
             "max_depth":[7,15,19,80,100,200],#***
             'class_weight':['balanced'],
             #'max_features':[5,7,9],#*
             #'min_samples_split':[110],#**
             #'min_samples_leaf':[80],#**
             #'random_state': [300]
             }              
        grid = GridSearchCV(rf, param_grid=param_grid, scoring='roc_auc', cv=10, verbose=0)#neg_log_loss,roc_auc
        grid.fit(x_train,y_train)
        y_pred=grid.predict(x_test)   
                        
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs, average='micro'
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(grid, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) # 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)        
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0)) 

    print("-------------------------")           
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("RF_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in RF_Res.csv")
    print("****************************************************")   
    
#------------------------------------------------------------------------------  
#--- Extreme Gradient Boosting  
#------------------------------------------------------------------------------  
def test_XGBoost(num_iters, features, labels):    
    
    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters) 
    
    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()         
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels) 
        xg=XGBClassifier(objective='binary:logistic' )
        PARAMETERS = {                      
                      "subsample":[0.9],
                      "colsample_bytree":[ 0.1],                
                      "max_depth":[7],
                       "eta":[0.1],
                      "min_child_weight":[1],
                      "learning_rate":[0.1,1],
                      "n_estimators":[1000],
                       "gamma":[0.1],
                        "reg_alpha":[0.1],
                        "use_label_encoder":['False'],
                        #'class_weight':['balanced'],
                        #'eval_metric':['error'] 
                     }
        eval_set = [(x_train, y_train),(x_test,y_test)]             
        grid = GridSearchCV(xg, param_grid=PARAMETERS, scoring='roc_auc', cv=10, verbose=0)#neg_log_loss,roc_auc
        grid.fit(x_train,y_train,eval_set=eval_set,early_stopping_rounds=200,eval_metric="error",verbose=0)
        y_pred=grid.predict(x_test) 
                    
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
       
       #--Get PermutationImportance
        perm = PermutationImportance(grid, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) # 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)            
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0)) 

    print("-------------------------")           
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")      
    with open("XGBoost_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))
    print("Done! The result is saved in XGBoost_Res.csv")
    print("****************************************************")   
    
#------------------------------------------------------------------------------  
#---  Multi-layer Perceptron  
#------------------------------------------------------------------------------  
def test_MLP(num_iters, features, labels):  

    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters)     

    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()        
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels) 
        mlp = MLPClassifier()    
        param_grid = {
             "hidden_layer_sizes": [(100,), (100, 30),(150,100,50)],
             "solver": [ 'lbfgs'],
             "max_iter": [110,360,400],
             'validation_fraction': [0.1],
             #'random_state': [1, 3, 5, 50],
             'activation': ['identity'],
             #'alpha': [0.0001, 0.05],
             #'class_weight':['balanced'],
             'learning_rate': ['adaptive']
             }              
        grid = GridSearchCV(mlp, param_grid=param_grid, scoring='accuracy', cv=10, verbose=0)#neg_log_loss,roc_auc
        grid.fit(x_train,y_train)
        y_pred=grid.predict(x_test) 
            
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(grid, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) # 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)        
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0))         

    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))   
    print("-------------------------")   
    with open("MLP_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))     
        print("Done! The result is saved in MLP_Res.csv")
    print("****************************************************")   
    
#------------------------------------------------------------------------------  
#---  Bayesian Additive Regression Trees  
#------------------------------------------------------------------------------  
def test_BART(num_iters, features, labels):   

    AUC_score = np.zeros(num_iters) 
    Accuracy_score = np.zeros(num_iters) 
    Precision_score = np.zeros(num_iters) 
    Recall_score = np.zeros(num_iters) 
    F1_score = np.zeros(num_iters)     

    for i in range(num_iters):
    
        #--get splits and best params
        np.random.RandomState(seed=1)
        t0 = time.time()        
        x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.1,random_state=i,stratify=labels)         
        model = SklearnModel() # Use default parameters
        model.fit(x_train, y_train) # Fit the model
        y_pred = model.predict(x_test) # Make predictions on new data
                   
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(model, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) # 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)        
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0))    
        
    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("BART_Res.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))   #write multi-rows use: writerows
        writer.writerow(map(lambda x: x,  permFstd/num_iters))
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))         
    print("Done! The result is saved in BART_Res.csv")
    print("****************************************************")   
    
if __name__ == '__main__':  
    
    pd.set_option("display.max_columns",50)

    #Set the number of iterations   
    items=100
    
    #Select the processing files 
    trian_data, test_data = load_csv_data('Sorted_cases_112.csv') 
    
    print("****************************************************")         
    print('LogisticRegression:\r')  
    test_LR(items, trian_data, test_data)   
     
    print('Support Vector Machine:\r')  
    test_SVM(items, trian_data, test_data) 
   
    print('Random Forest: \r')  
    test_RF(items, trian_data, test_data)  
        
    print('XGBoost: \r')  
    test_XGBoost(items, trian_data, test_data)
    
    print('Multi-layer Perceptron: \r')      
    test_MLP(items, trian_data, test_data) 
     
    print('Bayesian Additive Regression Trees: \r')  
    test_BART(items, trian_data, test_data)