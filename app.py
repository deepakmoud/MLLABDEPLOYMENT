from flask.helpers import send_file
from jinja2 import Template
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new"   
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew
#------------------------------ Saving dataset for Linear regression-------------------------------------------
# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot

#------------------------------ Saving image for K means-------------------------------------------
# this is the path to save figure of K menas
pathforelbowplot = "kmeans/plot"
#pathforonevarLRplot = "Regression/onevarLR/plot"
#app.config['LR1VAR'] = pathforonevarLR
app.config['elbowplot'] = pathforelbowplot
#print(app.config['elbowplot'])

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------------Data Preprocessing-------------------------------------------
# for data preprocessing

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing/preprocessing.html')


@app.route('/preprocessing/preprocessing' , methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        data_std = request.form['flexRadioDefault']
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dastaset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        
        # taking care of missing data
        imputer = SimpleImputer(missing_values=np.NAN, strategy='mean', fill_value=None, verbose=1, copy=True)
        imputer = imputer.fit(df.iloc[:, 0:col_no])
        df.iloc[:, 0:col_no] = imputer.transform(df.iloc[:, 0:col_no])

        # standerization
        
        if data_std == "yes":
            sc_X = StandardScaler()
            df = sc_X.fit_transform(df)
        trained_dataset = pd.DataFrame(df)
        trained_dataset.to_csv("static/data-preprocess/new/trained_dataset.csv")

        return render_template('/preprocessing/preprocessing_output.html', model_name=my_model_name, data_shape=trained_dataset.shape, table=trained_dataset.head(5).to_html(classes='table table-striped table-dark table-hover x'), dataset_describe=trained_dataset.describe().to_html(classes='table table-striped table-dark table-hover x'), )
#------------------------------Download Dataset-------------------------------------------
@app.route('/downloadNewDataset')
def download_file():
    path1 = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path1,as_attachment=True)

#------------------------------Download Model-------------------------------------------
@app.route('/downloadmodel')
def download_model():
    path1 = "static/data-preprocess/model/model.pkl"
    return send_file(path1,as_attachment=True)

#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Supervised machine Learning-------------------------------------------
# for suprvised learning

@app.route('/supervised')
def supervised():
    return render_template('/supervised/supervised.html')

#------------------------------Linear Regression-------------------------------------------
# for linear regression

@app.route('/supervised/regression/linearRegression')
def regressionLR():
    return render_template('/supervised/regression/linearRegression.html')


@app.route('/supervised/regression/linearRegression',  methods=['GET', 'POST'])
def simpleLinearRegression():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        intercept = regressor.intercept_
        slope = regressor.coef_
        training_score = regressor.score(X_train, y_train)
        testing_score = regressor.score(X_test, y_test)

        # visulization 
        plt.scatter(df.iloc[:,0],y, color='red')
        plt.plot(X, regressor.predict(X), color='blue')
        plt.title('{} VS {}'.format(li[0], li[1]))
        plt.xlabel('{}'.format(li[0]))
        plt.ylabel('{}'.format(li[1]))
        col_one = li[0]
        col_last = li[1]
        fig = plt.gcf()
        img_name = 'data'
        fig.savefig('static/Regression/onevarLR/plot/data.png', dpi=1500)
        get_plot1 = os.path.join(app.config['LR1VARplot'], '%s.png' % img_name)
        print(get_plot1)
        plt.clf()
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(regressor,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/regression/outputSimLR.html', dataset_name=my_dataset.filename, 
                               model_name=my_model_name,var1=intercept,var2=slope, visualize=get_plot1,
                               data_shape=df.shape, 
                               table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'), trainingscore=training_score, testingscore=testing_score, first_col=col_one, sec_col=col_last,)

  #------------------------------Linear Regression for prediction(specific dataset)-------------------------------------------
     
@app.route('/supervised/regression/outputSimLR',  methods=['GET', 'POST'])
def simpleLinearRegressionPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        xPred = float(num1)
        my_dataset = request.form['my_dataset']
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        X = df.iloc[:, :-1]
        y = df.iloc[:, 1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)
        
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        yPred = regressor.predict([[xPred]])
        yPred = float(yPred)
        col_one = li[0]
        secCol = li[-1]
        img_name = 'data'
        get_plot1 = os.path.join(app.config['LR1VARplot'], '%s.png' % img_name)
        return render_template('/supervised/regression/SLRpredicted.html', dataset_name=my_dataset, ans=yPred, model_name='Simple Linear Regressor Prediction', first_col=col_one, sec_col=secCol, num=num1, visualize=get_plot1)

#------------------------------Multiple Linear Regression-------------------------------------------
# for linear regression
# linear regression multi variable

@app.route('/supervised/regression/linearRegressionMV')
def regressionMVLR():
    return render_template('/supervised/regression/linearRegressionMV.html')

#-------------------------------Logistic Classification-------------------------------------------


@app.route('/supervised/logisticregression/logisticregression')
def logisticregression1():
    return render_template('/supervised/logisticregression/logisticregression.html')


@app.route('/supervised/logisticregression/logisticregression',  methods=['GET', 'POST'])
def logisticregression():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        print(class_type)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Logistic  Classification to the Training set
        from sklearn.linear_model  import LogisticRegression
        classifier = LogisticRegression(random_state=0,max_iter=1000)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type=="binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/logisticregression/logisticregressionoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Logistic Classification prediction(Poecific dataset)-------------------------------------------
        
@app.route('/supervised/logisticregression/logisticregressionoutput',  methods=['GET', 'POST'])
def logisticregressionPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Logistic Classification to the Training set
        from sklearn.linear_model  import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/logisticregression/logisticregressionpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Logistic Classifier', first_col=num1, second_col=num2, third_col=prediction)


#---------------------Decision Tree----------------------------------------------



@app.route('/supervised/decisiontree/tree')
def decisiontree1():
    return render_template('/supervised/decisiontree/tree.html')


@app.route('/supervised/decisiontree/tree',  methods=['GET', 'POST'])
def decisiontree():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        print(class_type)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "multiclass":
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            score = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/decisiontree/decisiontreeoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#---------------------Decision Tree prediction(Specific dataset)----------------------------------------------        
@app.route('/supervised/decisiontree/decisiontreeoutput',  methods=['GET', 'POST'])
def decisiontreePred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Decision Tree Classification to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/decisiontree/decisiontreepredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Decision Tree', first_col=num1, second_col=num2, third_col=prediction)

#---------------------Naive   Bayes Classification----------------------------------------------   


@app.route('/supervised/naivebayes/naivebayes')
def naivebayes1():
    return render_template('/supervised/naivebayes/naivebayes.html')


@app.route('/supervised/naivebayes/naivebayes',  methods=['GET', 'POST'])
def naivebayes():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        print(my_model_name)
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        print(data_std)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type =="binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/naivebayes/naivebayesoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Naive Bayes prediction(Specific dataset)-------------------------------------------
        
@app.route('/supervised/naivebayes/naivebayesoutput',  methods=['GET', 'POST'])
def naivebayesPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
       
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/naivebayes/naivebayespredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Naive Bayes Classifier', first_col=num1, second_col=num2, third_col=prediction)
#---------------------Random Forest Classification----------------------------------------------   



@app.route('/supervised/randomforest/randomforest')
def randomforest1():
    return render_template('/supervised/randomforest/randomforest.html')


@app.route('/supervised/randomforest/randomforest',  methods=['GET', 'POST'])
def randomforest():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        # Fitting Random Forest to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
        
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/randomforest/randomforestoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Random Forest Classification prediction(Specific dataset)-------------------------------------------
        
@app.route('/supervised/randomforest/randomforestoutput',  methods=['GET', 'POST'])
def randomforestPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
       
        # Fitting Naive Bayes to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/randomforest/randomforestpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Random forest Classifier', first_col=num1, second_col=num2, third_col=prediction)



#---------------------Support Vector Machine  Classification----------------------------------------------   



@app.route('/supervised/svm/svm')
def svm1():
    return render_template('/supervised/svm/svm.html')


@app.route('/supervised/svm/svm',  methods=['GET', 'POST'])
def svm():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0, probability=True)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/svm/svmoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Support Vector Machine Classification prediction(Specific dataset)-------------------------------------------
        
@app.route('/supervised/svm/svmoutput',  methods=['GET', 'POST'])
def svmPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
       
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0, probability=True)
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/svm/svmpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Support Vector Machine', first_col=num1, second_col=num2, third_col=prediction)

#---------------------k Nearest Neighbor  Classification----------------------------------------------   



@app.route('/supervised/knn/knn')
def knn1():
    return render_template('/supervised/knn/knn.html')


@app.route('/supervised/knn/knn',  methods=['GET', 'POST'])
def knn():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        #col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        
        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier =  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/supervised/knn/knnoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------KNN Classification prediction(Specific dataset)-------------------------------------------
        
@app.route('/supervised/knn/knnoutput',  methods=['GET', 'POST'])
def knnPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
       
       # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier =  KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
        if data_std == "yes":
            output = classifier.predict(sc.transform([[num1,num2]]))
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/supervised/knn/knnpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='K Nearest Neighbor', first_col=num1, second_col=num2, third_col=prediction)
#----------------------------------------------unsupervised learning------------------------------------




@app.route('/unsupervised')
def unsupervised():
    return render_template('/unsupervised/unsupervised.html')

#---------------------k Means Clustering----------------------------------------- 



@app.route('/unsupervised/kmeans/kmeans')
def kmeans1():
    return render_template('/unsupervised/kmeans/kmeans.html')


@app.route('/unsupervised/kmeans/kmeans',  methods=['GET', 'POST'])
def kmeans():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        ncluster = int(request.form['ncluster'])
        print(ncluster)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
       
        

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)
            
        # Using the elbow method to find the optimal number of clusters
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)

        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        fig = plt.gcf()
        img_name1 = 'elbow'
        fig.savefig('static/kmeans/plot/elbow.png', dpi=1500)
        #elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        elbow_plot = os.path.join('kmeans\plot', '%s.png' % img_name1)
        plt.clf()
        # Fitting K-Means to the dataset
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters =ncluster, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)
        var1=kmeans.inertia_
        var2=kmeans.cluster_centers_ 
        # Visualising the clusters
        plt.scatter(X[:,0], X[:,1], s = 100, c = 'black', label = 'Data Distribution')
        plt.title('Data Distribution before clustering')
        plt.xlabel('First feature ')
        plt.ylabel('Second Feature ')
        plt.legend()
        fig = plt.gcf()
        img_name2 = 'before'
        fig.savefig('static/kmeans/plot/before.png', dpi=1500)
        #before_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name2)
        before_plot = os.path.join('kmeans\plot', '%s.png' % img_name2)
        plt.clf()
        # Visualising the clusters
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_kmeans== 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('Data Distribution after clustering')
        plt.xlabel('first Feature')
        plt.ylabel('second feature')
        plt.legend()
        fig = plt.gcf()
        img_name3 = 'after'
        fig.savefig('static/kmeans/plot/after.png', dpi=1500)
        #f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        after_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name3)
        print(after_plot)
        plt.clf()
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(kmeans,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/unsupervised/kmeans/kmeansoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=var1,
                               var2=var2, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), visualize1=elbow_plot,visualize2=before_plot, 
                               visualize3=after_plot, dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Kmeans clustering prediction(Specific dataset)-------------------------------------------
        
@app.route('/unsupervised/kmeans/kmeansoutput',  methods=['GET', 'POST'])
def kmeansPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
        

        # Feature Scaling
        if data_std == "yes":
            # standardizing the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
       # Fitting K-Means to the dataset
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)
         
        predict= kmeans.predict([[num1,num2]])
        print(predict)
        if predict==[0]:
            result="Customer is careless"

        elif predict==[1]:
            result="Customer is standard"
        elif predict==[2]:
           result="Customer is Target"
        elif predict==[3]:
            result="Customer is careful"

        else:
            result="Custmor is sensible" 
       
        return render_template('/unsupervised/kmeans/kmeanspredicted.html', dataset_name=my_dataset, ans=predict, model_name='K Means Clustering', first_col=result)


#----------------------------------Hierarchical Clustering---------------------------------------------------


@app.route('/unsupervised/hierarchical/hierarchical')
def hierarchical1():
    return render_template('/unsupervised/hierarchical/hierarchical.html')


@app.route('/unsupervised/hierarchical/hierarchical',  methods=['GET', 'POST'])
def hierarchical():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        ncluster = int(request.form['ncluster'])
        print(ncluster)
        
        
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
       
        

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)
            
        # Using the dendrogram to find the optimal number of clusters
        from matplotlib import pyplot as plt
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        

        
        fig = plt.gcf()
        img_name1 = 'dendogram'
        fig.savefig('static/kmeans/plot/dendogram.png', dpi=1500)
        #elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        plt.clf()
        # Fitting Hierarchical Clustering to the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = ncluster, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
        from scipy.cluster.hierarchy import dendrogram, linkage
        import numpy as np
        Z = linkage(X, 'ward')
        from scipy.cluster.hierarchy import cophenet
        from scipy.spatial.distance import pdist

        c, coph_dists = cophenet(Z, pdist(X))
        var1= c
        var2=coph_dists 
        # Visualising the clusters
        plt.scatter(X[:,0], X[:,1], s = 100, c = 'black', label = 'Data Distribution')
        plt.title('Data Distribution before clustering')
        plt.xlabel('First feature ')
        plt.ylabel('Second Feature ')
        plt.legend()
        fig = plt.gcf()
        img_name2 = 'beforehierarchical'
        fig.savefig('static/kmeans/plot/beforehierarchical', dpi=1500)
        #before_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name2)
        before_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name2)
        plt.clf()
        # Visualising the clusters
        plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
        plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
        plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
        plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
        plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
        plt.title('Data Distribution after clustering')
        plt.xlabel('first Feature')
        plt.ylabel('second feature')
        plt.legend()
        fig = plt.gcf()
        img_name3 = 'afterhierarchical'
        fig.savefig('static/kmeans/plot/afterhierarchical.png', dpi=1500)
        #f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        after_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name3)
        print(after_plot)
        plt.clf()
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(kmeans,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/unsupervised/hierarchical/hierarchicaloutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=var1,
                               var2=var2, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), visualize1=elbow_plot,visualize2=before_plot, 
                               visualize3=after_plot, dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Hierarchical clustering prediction(Specific dataset)-------------------------------------------
        
@app.route('/unsupervised/hierarchical/hierarchicaloutput',  methods=['GET', 'POST'])
def hierarchicalPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
        

        # Feature Scaling
        if data_std == "yes":
            # standardizing the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
       # Fitting Hierarchical Clustering to the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
         
        predict= hc.predict([[num1,num2]])
        print(predict)
        if predict==[0]:
            result="Customer is careless"

        elif predict==[1]:
            result="Customer is standard"
        elif predict==[2]:
           result="Customer is Target"
        elif predict==[3]:
            result="Customer is careful"

        else:
            result="Custmor is sensible" 
       
        return render_template('/unsupervised/hierarchical/hierarchicalpredicted.html', dataset_name=my_dataset, ans=predict, model_name='hierarchical clustering', first_col=result)


#-------------------------DBSCAN CLUSTERING---------------------------------------------------------




@app.route('/unsupervised/dbscan/dbscan')
def dbscan1():
    return render_template('/unsupervised/dbscan/dbscan.html')


@app.route('/unsupervised/dbscan/dbscan',  methods=['GET', 'POST'])
def dbscan():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
       
        

        # Feature Scaling
        if data_std == "yes":
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X = sc.fit_transform(X)
            
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=3)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        var1=distances 
        distances = np.sort(distances, axis=0)
        distances1 = distances[:,1]
        
        from matplotlib import pyplot as plt
        plt.plot(distances1)
        plt.title('identification of optimal eps')
        plt.xlabel('data point indices')
        plt.ylabel('minimum distance')
        plt.legend()
        

        
        fig = plt.gcf()
        img_name1 = 'eps'
        fig.savefig('static/kmeans/plot/eps.png', dpi=1500)
        #elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        plt.clf()
        # define the model
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=5, min_samples=3, metric='euclidean')
        # fit model and predict clusters
        y_hc = model.fit(X)
        labels=model.labels_
        
        var2=model.core_sample_indices_
        var3=model.components_
        
        # Visualising the clusters
        plt.scatter(X[:,0], X[:,1], s = 100, c = 'black', label = 'Data Distribution')
        plt.title('Data Distribution before clustering')
        plt.xlabel('First feature ')
        plt.ylabel('Second Feature ')
        plt.legend()
        fig = plt.gcf()
        img_name2 = 'beforedbscan'
        fig.savefig('static/kmeans/plot/beforedbscan.png', dpi=1500)
        #before_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name2)
        before_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name2)
        plt.clf()
        y_pred = model.fit_predict(X)
        plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired', label = 'Clusters')
        plt.title('Students segmenation')
        plt.xlabel('Btech Aggregate')
        plt.ylabel('Final performance')
        plt.legend()
        
        plt.title('Data Distribution after clustering')
        plt.xlabel('first Feature')
        plt.ylabel('second feature')
        plt.legend()
        fig = plt.gcf()
        img_name3 = 'afterdbscan'
        fig.savefig('static/kmeans/plot/afterdbscan.png', dpi=1500)
        #f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        after_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name3)
        print(after_plot)
        plt.clf()
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(kmeans,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/unsupervised/dbscan/dbscanoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=var1,
                               var2=var2,var3=var3, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), visualize1=elbow_plot,visualize2=before_plot, 
                               visualize3=after_plot, dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------DBSCAN clustering prediction(Specific dataset)-------------------------------------------
        
@app.route('/unsupervised/dbscan/dbscanoutput',  methods=['GET', 'POST'])
def dbscanPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :].values
        

        # Feature Scaling
        if data_std == "yes":
            # standardizing the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
       # define the model
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=5, min_samples=3, metric='euclidean')
        # fit model and predict clusters
        y_hc = model.fit(X)
         
        predict= model.fit_predict([[num1,num2]])
        print(predict)
        if predict==[0]:
            result="Customer is careless"

        elif predict==[1]:
            result="Customer is standard"
        elif predict==[2]:
           result="Customer is Target"
        elif predict==[3]:
            result="Customer is careful"

        else:
            result="Custmor is sensible" 
       
        return render_template('/unsupervised/dbscan/dbscanpredicted.html', dataset_name=my_dataset, ans=predict, model_name='hierarchical clustering', first_col=result)


#----------------------------------------------Feature Engineering and Feature selection------------------------------------


@app.route('/feature')
def feature():
    return render_template('/feature/feature.html')



#----------------------------------------------PCA---------------------------------------

@app.route('/feature/pca/pca')
def pca1():
    return render_template('/feature/pca/pca.html')


@app.route('/feature/pca/pca',  methods=['GET', 'POST'])
def pca():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['classification']
        ncomponenets=int(request.form['ncomponents'])
        print(ncomponenets)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = None)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        var5=explained_variance
        # Applying PCA
        from sklearn.decomposition import PCA
        pca1 = PCA(n_components = ncomponenets)
        X_train = pca1.fit_transform(X_train)
        X_test = pca1.transform(X_test)
        explained_variance = pca1.explained_variance_ratio_
        var6=explained_variance
       # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(pca1,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/feature/pca/pcaoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,var5=var5, var6=var6, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Support Vector Machine Classification prediction(Specific dataset)-------------------------------------------
        
@app.route('/feature/pca/pcaoutput',  methods=['GET', 'POST'])
def pcaPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
       
        # Applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        explained_variance = pca.explained_variance_ratio_
        var6=explained_variance
       # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_train, y_train)
        
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/feature/pca/pcapredicted.html', dataset_name=my_dataset, ans=prediction, model_name='PCA', first_col=num1, second_col=num2, third_col=prediction)

#-------------------------------------------------------------------LDA----



@app.route('/feature/lda/lda')
def lda1():
    return render_template('/feature/lda/lda.html')


@app.route('/feature/lda/lda',  methods=['GET', 'POST'])
def lda():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['flexRadioDefault1']
        print(class_type)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        # Applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components =None)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        explained_variance = lda.explained_variance_ratio_
        var5=explained_variance
       
        if class_type == "multiclass":
            # Applying LDA
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            lda = LDA(n_components = 2)
            X_train = lda1.fit_transform(X_train, y_train)
            X_test = lda1.transform(X_test)
            var6=explained_variance
       
       # Applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 1)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
        var6=explained_variance
        
       # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        # Making the Confusion Matrix
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(lda,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/feature/lda/ldaoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,var5=var5, var6=var6, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Support Vector Machine Classification prediction(Specific dataset)-------------------------------------------
        
@app.route('/feature/lda/ldaoutput',  methods=['GET', 'POST'])
def ldaPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42)

        # Feature Scaling
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
       
        # Applying LDA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components = 2)
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)

       # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_train, y_train)
        
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/feature/lda/ldapredicted.html', dataset_name=my_dataset, ans=prediction, model_name='LDA', first_col=num1, second_col=num2, third_col=prediction)


#-------------------------------------Feature selection Filter Based Method----



@app.route('/feature/filter/filter')
def filter1():
    return render_template('/feature/filter/filter.html')


@app.route('/feature/filter/filter',  methods=['GET', 'POST'])
def filter2():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['flexRadioDefault1']
        selection_method=request.form['flexRadio']
        
        print(class_type)
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(df.drop(cols[-1], axis=1), 
                                                  df[cols[-1]], test_size=0.3, 
                                                 random_state=0)

        
        
        # Applying feature selection
        # import and create the VarianceThreshold object.
        from sklearn.feature_selection import VarianceThreshold
        vs_constant = VarianceThreshold(threshold=0)

        # select the numerical columns only.
        numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]

        # fit the object to our data.
        vs_constant.fit(numerical_x_train)

        # get the constant colum names.
        constant_columns = [column for column in numerical_x_train.columns
                    if column not in numerical_x_train.columns[vs_constant.get_support()]]

        # detect constant categorical variables.
        constant_cat_columns = [column for column in X_train.columns 
                        if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]

        # conctenating the two lists.
        all_constant_columns = constant_cat_columns + constant_columns
        var5=all_constant_columns
        # drop the constant columns
        X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
        X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
        threshold = 0.80

        # create empty list
        quasi_constant_feature = []

            # loop over all the columns
        for feature in X_train.columns:

            # calculate the ratio.
            predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
    
            # append the column name if it is bigger than the threshold
            if predominant >= threshold:
                quasi_constant_feature.append(feature)   
        
        var6= quasi_constant_feature
        # drop the quasi constant columns
        X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        
        # transpose the feature matrice
        train_features_T = X_train.T

        # print the number of duplicated features
        

        # select the duplicated features columns names
        duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
        var2=duplicated_columns
        # drop those columns
        X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
        X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
        
        # creating set to hold the correlated features
        corr_features = set()

        # create the correlation matrix (default to pearson)
        corr_matrix = X_train.corr()

        

        for i in range(len(corr_matrix .columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    colname = corr_matrix.columns[i]
                    corr_features.add(colname)

        var3=corr_features
        print(var3)            
        X_train.drop(labels=corr_features, axis=1, inplace=True)
        X_test.drop(labels=corr_features, axis=1, inplace=True)
        if selection_method == "pearson":
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_regression
            
            
            # define feature selection
            fs = SelectKBest(score_func=f_regression, k=4)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
           
            var4= X_selected[1].shape
            print(var4)
        elif selection_method == "annova":
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_classif
            # define feature selection
            fs = SelectKBest(score_func=f_classif, k=4)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
            var4= X_selected[1].shape
        else :
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import mutual_info_classif
            # define feature selection
            fs = SelectKBest(score_func=mutual_info_classif, k=4)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
           
            var4= X_selected[1].shape
            
       
        
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_selected, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        Accuracy= accuracy_score(y_test, y_pred)*100
        #print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/feature/filter/filteroutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=var2,var3=var3, var4=var4,var5=var5, var6=var6, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#------------------------------- Classification after feature selection prediction(Specific dataset)-------------------------------------------
        
@app.route('/feature/filter/filteroutput',  methods=['GET', 'POST'])
def filterPred():
    if request.method == 'POST':
        num1 = request.form['num1']
        num2 = request.form['num2']
        
        my_dataset = request.form['my_dataset']
        data_std = request.form['flexRadioDefault']
        selection_method=request.form['flexRadio']
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        li = list(df.columns)
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(df.drop(cols[-1], axis=1), 
                                                  df[cols[-1]], test_size=0.3, 
                                                 random_state=0)

        
        
        # Applying feature selection
        # import and create the VarianceThreshold object.
        from sklearn.feature_selection import VarianceThreshold
        vs_constant = VarianceThreshold(threshold=0)

        # select the numerical columns only.
        numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]

        # fit the object to our data.
        vs_constant.fit(numerical_x_train)

        # get the constant colum names.
        constant_columns = [column for column in numerical_x_train.columns
                    if column not in numerical_x_train.columns[vs_constant.get_support()]]

        # detect constant categorical variables.
        constant_cat_columns = [column for column in X_train.columns 
                        if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]

        # conctenating the two lists.
        all_constant_columns = constant_cat_columns + constant_columns
        var5=all_constant_columns
        # drop the constant columns
        X_train.drop(labels=all_constant_columns, axis=1, inplace=True)
        X_test.drop(labels=all_constant_columns, axis=1, inplace=True)
        threshold = 0.80

        # create empty list
        quasi_constant_feature = []

            # loop over all the columns
        for feature in X_train.columns:

            # calculate the ratio.
            predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
    
            # append the column name if it is bigger than the threshold
            if predominant >= threshold:
                quasi_constant_feature.append(feature)   
        
        var6= quasi_constant_feature
        # drop the quasi constant columns
        X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
        
        # transpose the feature matrice
        train_features_T = X_train.T

        # print the number of duplicated features
        

        # select the duplicated features columns names
        duplicated_columns = train_features_T[train_features_T.duplicated()].index.values
        var2=duplicated_columns
        # drop those columns
        X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
        X_test.drop(labels=duplicated_columns, axis=1, inplace=True)
        
        # creating set to hold the correlated features
        corr_features = set()

        # create the correlation matrix (default to pearson)
        corr_matrix = X_train.corr()

        

        for i in range(len(corr_matrix .columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    colname = corr_matrix.columns[i]
                    corr_features.add(colname)

        var3=corr_features
        print(var3)            
        X_train.drop(labels=corr_features, axis=1, inplace=True)
        X_test.drop(labels=corr_features, axis=1, inplace=True)
        if selection_method == "pearson":
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_regression
            
            
            # define feature selection
            fs = SelectKBest(score_func=f_regression, k=2)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
           
            var4= X_selected[1].shape
            
        elif selection_method == "annova":
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_classif
            # define feature selection
            fs = SelectKBest(score_func=f_classif, k=2)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
            var4= X_selected[1].shape
        else :
            # pearson's correlation feature selection for numeric input and numeric output
            from sklearn.datasets import make_regression
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import mutual_info_classif
            # define feature selection
            fs = SelectKBest(score_func=mutual_info_classif, k=2)
            # apply feature selection
            X_selected = fs.fit_transform(X_train, y_train)
            X_test= fs.transform(X_test)
           
            var4= X_selected[1].shape
            
       
        
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        classifier.fit(X_selected, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        output = classifier.predict([[num1,num2]])
        if output==[1]:
            prediction="Item will be purchased"
        else:
            prediction="Item will not be purchased"
        
       
        return render_template('/feature/filter/filterpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Classification based on Feature selection', first_col=num1, second_col=num2, third_col=prediction)



#-----------------Feature selection Wrapper method-----------------------------------

@app.route('/feature/wrapper/wrapper')
def wrapper1():
    return render_template('/feature/wrapper/wrapper.html')


@app.route('/feature/wrapper/wrapper',  methods=['GET', 'POST'])
def wrapper2():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        data_std = request.form['flexRadioDefault']
        class_type=request.form['flexRadioDefault1']
        selection_method=request.form['flexRadio']
        
        
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dataset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        #df = df.fillna(method='ffill')
        #li = list(df.columns)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)
        from mlxtend.feature_selection import SequentialFeatureSelector

        # import the algorithm you want to evaluate on your features.
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_regression
        
        
        if selection_method == "forward":
            # create the SequentialFeatureSelector object, and configure the parameters.
            sfs = SequentialFeatureSelector(RandomForestClassifier(), 
                                            k_features=5, 
                                            forward=True, 
                                            floating=False,
                                            scoring='accuracy',
                                            cv=2)

            # fit the object to the training data.
            sfs = sfs.fit(X_train, y_train)

            # print the selected features.
            selected_features = X_train[list(sfs.k_feature_idx_)]
            var5= sfs.k_feature_names_

            # print the final prediction score.
            var6= sfs.k_score_

            # transform to the newly selected features.
            x_train_sfs = sfs.transform(X_train)
            import pickle 
            # Save the trained model as a pickle string. 
            saved_model=pickle.dump(sfs,open("static/data-preprocess/model/model.pkl", 'wb'))

        elif selection_method == "backward":
            # create the SequentialFeatureSelector object, and configure the parameters.
            sfs = SequentialFeatureSelector(RandomForestClassifier(), 
                                            k_features=5, 
                                            forward=False, 
                                            floating=False,
                                            scoring='accuracy',
                                            cv=2)

            # fit the object to the training data.
            sfs = sfs.fit(X_train, y_train)

            # print the selected features.
            selected_features = X_train[list(sfs.k_feature_idx_)]
            var5= sfs.k_feature_names_

            # print the final prediction score.
            var6= sfs.k_score_

            # transform to the newly selected features.
            X_train_sfs = sfs.transform(X_train)
            import pickle 
            # Save the trained model as a pickle string. 
            saved_model=pickle.dump(sfs,open("static/data-preprocess/model/model.pkl", 'wb'))
            
        else :
            from mlxtend.feature_selection import ExhaustiveFeatureSelector
           
            # import the algorithm you want to evaluate on your features.
            from sklearn.ensemble import RandomForestClassifier

            # create the ExhaustiveFeatureSelector object.
            efs = ExhaustiveFeatureSelector(RandomForestClassifier(), 
                                            min_features=2,
                                            max_features=6, 
                                            scoring='roc_auc',
                                            cv=2)

            # fit the object to the training data.
            efs = efs.fit(X_train, y_train)

            # print the selected features.
            selected_features = X_train[list(efs.best_idx_ )]
            var5 = selected_features

            # print the final prediction score.
            var6= efs.best_score_

            # transform our data to the newly selected features.
            x_train_sfs = efs.transform(X_train)
            x_test_sfs = efs.transform(X_test)
            import pickle 
            # Save the trained model as a pickle string. 
            saved_model=pickle.dump(efs,open("static/data-preprocess/model/model.pkl", 'wb'))
           
        
        # Fitting SVM to the Training set
        #from sklearn.svm import SVC
        #classifier = SVC(kernel = 'sigmoid', random_state = 0,probability=True)
        #classifier.fit(x_train_sfs, y_train)
        # Predicting the Test set results
        #y_pred = classifier.predict(x_train_sfs)
        #Accuracy= accuracy_score(y_train, y_pred)*100
        #print(Accuracy)
        #from sklearn.metrics import precision_score
        #from sklearn.metrics import recall_score
        #from sklearn.metrics import f1_score
        #if class_type == "binary":
           # precision = precision_score(y_test, y_pred, average='binary')
           # recall = recall_score(y_test, y_pred, average='binary')
            #score = f1_score(y_test, y_pred, average='binary')
        #precision = precision_score(y_test, y_pred, average='macro')
        #recall = recall_score(y_test, y_pred, average='macro')
        #score = f1_score(y_test, y_pred, average='macro')
       # ) 
    
        return render_template('/feature/wrapper/wrapperoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,
                               var5=var5, var6=var6, data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))



#-----------------Aprori Algorithm-----------------------------------

@app.route('/unsupervised/apriori/apriori')
def apriori1():
    return render_template('/unsupervised/apriori/apriori.html')


@app.route('/unsupervised/apriori/apriori',  methods=['GET', 'POST'])
def apriori2():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        minimum_support= float(request.form['support'])
        minimum_confidence=float(request.form['confidence'])
        
        
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        # Importing the libraries
        import pandas as pd
        import numpy as np
        from mlxtend.frequent_patterns import apriori, association_rules
        import matplotlib.pyplot as plt
        dataset = pd.read_csv(get_dataset, nrows=1000)
        #Count total number of classes in Data
        items = (dataset['0'].unique())
        var1=items
        #Create list 
        transactions = []
        
        for i in range(0, dataset.shape[0]):
            transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])
        
        
        import pandas as pd
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        df=df.drop(['nan'], axis = 1)
        freq_items = apriori(df, min_support=minimum_support, use_colnames=True)
        var2=freq_items
        rules = association_rules(freq_items, metric="confidence", min_threshold=minimum_confidence)
        var3=rules
        import pickle 
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(freq_items,open("static/data-preprocess/model/model.pkl", 'wb'))
           
        
      
    
        return render_template('/unsupervised/apriori/apriorioutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,
                               var1=var1, var2=var2.to_html( classes='table table-striped table-dark table-hover x'),var3=var3.to_html( classes='table table-striped table-dark table-hover x'), data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))



#-----------------FP Growth Algorithm-----------------------------------

@app.route('/unsupervised/fp/fp')
def fp1():
    return render_template('/unsupervised/fp/fp.html')


@app.route('/unsupervised/fp/fp',  methods=['GET', 'POST'])
def fp():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        minimum_support= float(request.form['support'])
        minimum_confidence=float(request.form['confidence'])
        
        
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        # Importing the libraries
        import pandas as pd
        import numpy as np
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.frequent_patterns import fpgrowth
        import matplotlib.pyplot as plt
        dataset = pd.read_csv(get_dataset,nrows=1000)
        #Count total number of classes in Data
        items = (dataset['0'].unique())
        var1=items
        #Create list 
        transactions = []
        
        for i in range(0, dataset.shape[0]):
            transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])])
        
        
        import pandas as pd
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        df=df.drop(['nan'], axis = 1)
        freq_items = fpgrowth(df, min_support=minimum_support, use_colnames=True)
        var2=freq_items
        rules = association_rules(freq_items, metric="confidence", min_threshold=minimum_confidence)
        var3=rules
        import pickle 
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(freq_items,open("static/data-preprocess/model/model.pkl", 'wb'))
           
        
      
    
        return render_template('/unsupervised/fp/fpoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,
                               var1=var1, var2=var2.to_html( classes='table table-striped table-dark table-hover x'),var3=var3.to_html( classes='table table-striped table-dark table-hover x'), data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))




#-----------------Natural language processing-----------------------------------

@app.route('/nlp')
def nlp():
    return render_template('/nlp/nlp.html')


#----------------------Sentiment Analysis------------------------------


@app.route('/nlp/sentiment/sentiment')
def sentiment1():
    return render_template('/nlp/sentiment/sentiment.html')


@app.route('/nlp/sentiment/sentiment',  methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model1']
        
        class_type=request.form['classification']
        dataset_path = os.path.join(pathforonevarLR, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset.filename))
        input=secure_filename(my_dataset.filename)
        extension= input.split(".")
        extension=extension[1]
        print(extension)
        if extension == "csv":
            df = pd.read_csv(get_dataset)
        else:
            #target=df.iloc[:,-1].values
            #print(target)
            df = pd.read_csv(get_dataset, delimiter="\t", quoting=3,nrows=1000)
        # Cleaning the texts for all review using for loop
        df = df.dropna()
        #target=df.iloc[:,-1].values
        #df.reset_index(inplace = True) 
        #df = df.drop(['index'], axis = 1) 
        import re
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, df.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', df.iloc[:,0][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
       
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)

        X = cv.fit_transform(corpus).toarray()
        
        y = df.iloc[:,-1].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        

        
        
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
           
        
        
        Accuracy= accuracy_score(y_test, y_pred)*100
        print(Accuracy)
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score
        if class_type == "binary":
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            score = f1_score(y_test, y_pred, average='binary')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        score = f1_score(y_test, y_pred, average='macro')
        import pickle 
        print("[INFO] Saving model...")
        # Save the trained model as a pickle string. 
        saved_model=pickle.dump(classifier,open("static/data-preprocess/model/model.pkl", 'wb')) 
    
        return render_template('/nlp/sentiment/sentimentoutput.html', dataset_name=my_dataset.filename, model_name=my_model_name,var1=Accuracy,
                               var2=precision,var3=recall, var4=score,  data_shape=df.shape, table=df.head(5).to_html( classes='table table-striped table-dark table-hover x'), dataset_describe=df.describe().to_html(classes='table table-striped table-dark table-hover x'))

#-------------------------------Sentiment Analysis prediction(Specific dataset)-------------------------------------------
        
@app.route('/nlp/sentiment/sentimentoutput',  methods=['GET', 'POST'])
def sentimentPred():
    if request.method == 'POST':
        review_input = request.form['review']
       
        
        my_dataset = request.form['my_dataset']
        
       
        get_dataset = os.path.join(app.config['LR1VAR'], secure_filename(my_dataset))
        df = pd.read_csv(get_dataset, delimiter="\t", quoting=3)
        # Cleaning the texts for all review using for loop
        
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        corpus = []
        for i in range(0, df.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
    
            corpus.append(review)#df = df.fillna(method='ffill')
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)

        X = cv.fit_transform(corpus).toarray()
        y = df.iloc[:, 1].values
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)#col_no = df.shape[1]

        

        
        
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        
        #output = classifier.predict([[review]])
        input_data = [review_input ] 
  
        input_data = cv.transform(input_data).toarray()


        input_pred = classifier.predict(input_data)

        input_pred = input_pred.astype(int)


        if input_pred[0]==1:
            prediction=" Positive"
        else:
            prediction=" Negative"
        
        
       
        return render_template('/nlp/sentiment/sentimentpredicted.html', dataset_name=my_dataset, ans=prediction, model_name='Sentiment Analysis', first_col=review_input, second_col=prediction)
#---------------Application----------------
if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"