# Prêt-à-dépenser : Streamlit Dashboard

This dashboard has been deployed on : https://dubois-credit-frontend-home.streamlit.app/


# Dashboard Content

The dashboard contains on Home page and 3 other pages :

## Home page 

The home page contains two majors components :
* The presentation of the dashboard and main contents of the app
* The loading of the data, the model and computing the SHAP explainer

/!\ IMPORTANT : Please wait for the end of the loading before changing pages. Datas and model need to be registered to the cache for the other pages. 

## 1st page : General information

This page contains 3 tabs describing the data and the model 
* The first tab presents the number of clients, number of features, the target distribution and informations about missing values
* The second tab presents plots for each of the 62 features : a Bar plot of frequency for binary features, a box plot for quantitative features. 
* The third tab presents the model, the feature importance, its best parameters, scores and plots on a validation set. 

## 2nd page : Know clients

 
**Input:** The input category takes the client ID as an argument. The officer needs to know in advance the unique ID of the client. It returns an error in case of incorrect ID

The app first shows the target of the client, the predicted target by the model and its probability.\ 
Then, a plot of SHAP values is generated (and recoloried). From this graphs, the 6 top discriminative features are represented to compare the specific client to others. 

/!\ IMPORTANT : This page does NOT use an external API but the model loaded previously.

## 3rd page : Prediction API

This page use an external API to predict the loan difficulties for new clients.\ 
The external API is build with FASTAPI, and return the prediction of loan default and its probability (with a decision threshold set at 0.4).

This page contains two tabs :
* The first tab is for clients with an ID (and data) but with unknown target. It uses the Client ID as an input.
* The second tab is for new datas : The officer can enter data with three different manners: Manually with one box per features, using a text box with a pre-load dictionnary or uploading a csv file. 


In both tabs, the dashboard returns the same output (prediction and plots) as the known clients page. 






 
