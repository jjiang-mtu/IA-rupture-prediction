import pandas as pd
import numpy as np 
import time
import csv
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import eli5   # python permutation importance tools
from eli5.sklearn import PermutationImportance

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from bartpy.sklearnmodel import SklearnModel

import lime
import lime.lime_tabular
import shap

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
        model = sklearn.linear_model.LogisticRegression(solver='liblinear',penalty="l2", C=0.1)
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
            
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs
        Precision_score[i] = precision_score(y_test,y_pred)
        Recall_score[i] = recall_score(y_test,y_pred)
        F1_score[i] = f1_score(y_test,y_pred)
        
        #--Get PermutationImportance
        perm = PermutationImportance(model, random_state = 1,n_iter=5,scoring='accuracy').fit(x_test, y_test) 
        if i==0:
            permFea = np.abs(perm.feature_importances_)
            permFstd = np.abs(perm.feature_importances_std_)
        else:
            permFea += np.abs(perm.feature_importances_)
            permFstd += np.abs(perm.feature_importances_std_)
                    
        t1 = time.time()
        print("Iteration:", i, "Accuracy_Score:%.4f" % float(Accuracy_score[i]), "Predictions:", y_pred, "Time_cost:%.4f" % float(t1-t0)) 
        
        #--Get LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], model.predict_proba, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.Explainer(model, x_test)
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
            
    print("-------------------------")               
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("LR_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in LR_LIME_SHAP.csv")
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
        model = svm.SVC(kernel='linear',class_weight='balanced',probability=True)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
            
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
        
        #--Get LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], model.predict_proba, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.KernelExplainer(model.predict,x_test)
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
            
    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")      
    with open("SVM_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in SVM_LIME_SHAP.csv")
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
        model = RandomForestClassifier(max_depth=6, random_state=0, n_estimators=10)
        model.fit(x_train, y_train)    
        y_pred=model.predict(x_test)
                        
        #--Get ROC-AUC Score
        Accuracy_score[i] = accuracy_score(y_test,y_pred)        
        AUC_score[i] = roc_auc_score(y_test, y_pred) #--Choose macro, micro etc. See Docs, average='micro'
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
        
        #--Get LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], model.predict_proba, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.Explainer(model)
        shap_values = explainer(x_test)        
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
            
    print("-------------------------")           
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("RF_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score)) 
    print("Done! The result is saved in RF_LIME_SHAP.csv")
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
        model = XGBClassifier(objective='binary:logistic', colsample_bytree=0.1,eta=0.1).fit(x_train, y_train, eval_metric='rmse')
        y_pred=model.predict(x_test) 
                    
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
        
        #--Get LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], model.predict_proba, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.Explainer(model)
        shap_values = explainer(x_test)        
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
        
    print("-------------------------")           
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")      
    with open("XGBoost_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))
    print("Done! The result is saved in XGBoost_LIME_SHAP.csv")
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
        model = MLPClassifier(activation='identity',learning_rate='adaptive', solver='lbfgs')
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test) 
            
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
        
        #--Get LIME
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], model.predict_proba, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.KernelExplainer(model.predict, x_test)    
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
            
    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))   
    print("-------------------------")   
    with open("MLP_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))     
        print("Done! The result is saved in MLP_LIME_SHAP.csv")
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
        
        #--Get LIME
        def prob(data):
            return np.array(list(zip(1-model.predict(data),model.predict(data)))) 
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                    feature_names=x_test.columns.tolist(),
                                                   class_names=['Unruptured', 'Unruptured/Ruptured'], discretize_continuous=True)
        exp = explainer.explain_instance(x_test.values[1], prob, num_features=34)
        limeFea = np.abs(sorted(exp.as_map()[1]))[:,1]
        if i==0:
            limeFea = np.abs(limeFea)
        else:
            limeFea += np.abs(limeFea)   
            
        #--Get SHAP            
        explainer = shap.KernelExplainer(model.predict, x_test)
        cols =x_test.columns.tolist()
        shap_values_f = explainer.shap_values(x_test)
        if i==0:
            shapFea = np.mean(np.abs(shap_values_f), axis = 0)
        else:
            shapFea += np.mean(np.abs(shap_values_f), axis = 0)
            
    print("-------------------------")   
    print("Accuracy Score: %.4f" % np.mean(Accuracy_score))
    print("AUC Score: %.4f" % np.mean(AUC_score))
    print("Precision Score: %.4f" % np.mean(Precision_score))
    print("Recall Score: %.4f" % np.mean(Recall_score))
    print("F1 Score: %.4f" % np.mean(F1_score))
    print("-------------------------")   
    with open("BART_LIME_SHAP.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(x_test.columns.tolist())   # write columns_name
        writer.writerow(map(lambda x: x,  permFea/num_iters))  
        writer.writerow(map(lambda x: x,  limeFea/num_iters)) 
        writer.writerow(map(lambda x: x,  shapFea/num_iters)) 
        writer.writerow(map(lambda x: x,  Accuracy_score)) 
        writer.writerow(map(lambda x: x,  AUC_score))   
        writer.writerow(map(lambda x: x,  Precision_score))  
        writer.writerow(map(lambda x: x,  Recall_score)) 
        writer.writerow(map(lambda x: x,  F1_score))         
    print("Done! The result is saved in BART_LIME_SHAP.csv")
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