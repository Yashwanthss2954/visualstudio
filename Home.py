import streamlit as st
import joblib
import zipfile
import time
import altair as alt
#from streamlit_lottie import st_lottie
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, load_linnerud
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix, hamming_loss, jaccard_score, log_loss, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, roc_curve, zero_one_loss
from sklearn.model_selection import train_test_split
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    explained_variance_score
)


from sklearn.linear_model._glm.glm import GammaRegressor
from sklearn.ensemble._gb import GradientBoostingRegressor
from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model._coordinate_descent import ElasticNet
from sklearn.neural_network._multilayer_perceptron import MLPRegressor
from sklearn.linear_model._stochastic_gradient import SGDRegressor
from algorithms.classifiers import *
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

requirements_txt_file = """
fastapi
numpy
"""

backend_api_python_program_content = """

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Define a request model
class PredictionRequest(BaseModel):
    # Sample POST Request: 
    # {
    #     "features": [5.1, 3.5, 1.4, 0.2]
    # }
    features: list[float]

# Initialize FastAPI app
app = FastAPI()

# Load the machine learning model
model = None

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("model.joblib")

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert request features to a numpy array and reshape it for prediction
    features = np.array(request.features).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(features)

    # Return the prediction result
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""


def create_zip(model_buffer):

    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w') as zip_file:

        zip_file.writestr('backend_api.py', backend_api_python_program_content)

        zip_file.writestr('model.joblib', model_buffer.getvalue())

        zip_file.writestr('requirements.txt', requirements_txt_file)


    buffer.seek(0)
    return buffer



def main():
    st.markdown("<center><h1>Model Studio</h1></center>", unsafe_allow_html=True)
    #st.lottie("https://lottie.host/f9ecc8cd-9a0e-49f5-bfbe-89bb59ca794b/Qnv20SfUVi.json", height=50, width=50, quality="high")
    # st.markdown("<center><h4><b>By Metric Coders</b></h4></center>", unsafe_allow_html=True)
    st.markdown("<center><h4><b>A No-Code Platform to train and deploy your Large Language Models</b></h4></center>", unsafe_allow_html=True)
    dataset = load_iris()
    clf = LogisticRegression()
    regr = LinearRegression()
    X = None
    y = None
    ml_algo_options = ["Classifiers", "Regressors"]

    algo_type = st.radio("Select the type of ML algorithm:", ml_algo_options)

    own_dataset = st.checkbox(label="Load Own Dataset (The CSV file should have the header by the name 'target' as the result column)", value=False)
    if own_dataset:
        uploaded_file = st.file_uploader("Upload Training Data in a CSV file", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            st.write("Preview of Dataset")
            st.write(data.head())

            X = data.drop("target", axis=1)
            y = data["target"]
    else:
        if algo_type == "Classifiers":
            dataset_option = st.selectbox("Dataset", ["Iris", "Digits", "Wine", "Breast Cancer"], index=0)
            if dataset_option == "Iris":
                dataset = load_iris()
            elif dataset_option == "Digits":
                dataset = load_digits()
            elif dataset_option == "Wine":
                dataset = load_wine()
            elif dataset_option == "Breast Cancer":
                dataset = load_breast_cancer()
        elif algo_type == "Regressors":
            dataset_option = st.selectbox("Dataset", ["Diabetes", "Linnerrud"], index=0)
            if dataset_option == "Diabetes":
                dataset = load_diabetes()
            elif dataset_option == "Linnerrud":
                dataset = load_linnerud()

        X = dataset.data
        y = dataset.target

    if algo_type == "Classifiers":
        ml_algorithm = st.selectbox("Classifiers", [
                                                 "Decision Tree Classifier",
                                                 "Gradient Boosting Classifier",
                                                 "KNeighbors Classifier",
                                                 "Random Forest Classifier",
                                                 "One Class SVM",
                                                 "Logistic Regression"
                                                 ], index=0)
    else:
        ml_algorithm = st.selectbox("Regressors", [
            "Elastic Net",
            "Gamma Regressor",
            "Gradient Boosting Regressor",
            "SGD Regressor",
            "MLP Regressor",
            "Linear Regression"
        ], index=0)
    col1, col2 = st.columns(2)

    col1.markdown("<center><h3>Hyperparameters</h3></center>" ,unsafe_allow_html=True)
    col2.markdown("<center><h3>Data Visualization</h3></center>" ,unsafe_allow_html=True)
    col2.dataframe(pd.DataFrame(X), use_container_width=True)

    # col2.markdown("<center><h3>Target Visualization</h3></center>", unsafe_allow_html=True)
    # col2.dataframe(pd.DataFrame(y), use_container_width=True)
    if not own_dataset:
        chart_data = pd.DataFrame(X, columns=dataset.feature_names)

        col2.markdown("<center><h3>Scatter Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.scatter_chart(chart_data)

        col2.markdown("<center><h3>Bar Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.bar_chart(chart_data)

        col2.markdown("<center><h3>Line Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.line_chart(chart_data)

    else:
        chart_data = pd.DataFrame(X, columns=y)

        col2.markdown("<center><h3>Scatter Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.scatter_chart(chart_data)

        col2.markdown("<center><h3>Bar Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.bar_chart(chart_data)

        col2.markdown("<center><h3>Line Chart Visualization</h3></center>", unsafe_allow_html=True)

        col2.line_chart(chart_data)


    if ml_algorithm == "KNeighbors Classifier":
        n_neighbors = col1.slider("n_neighbors", min_value=1, max_value=100, value=5, step=1)
        weights = col1.selectbox("weights", ["uniform", "distance"], index=0)
        algorithm = col1.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0)

        leaf_size = col1.slider("leaf_size", min_value=1, max_value=100, value=30, step=1)
        n_jobs = col1.slider("n_jobs", min_value=1, max_value=100, value=5, step=1)

        clf = get_kneighbors(n_neighbors, weights, algorithm, leaf_size, n_jobs)



    elif ml_algorithm == "Decision Tree Classifier":
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion", index=0)
        splitter = col1.selectbox("splitter", ["best", "random"], index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                                      index=2)
        ccp_alpha = col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)

        clf = get_decision_tree(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha
        )


    elif ml_algorithm == "Gradient Boosting Classifier":
        n_estimators = col1.slider("n_estimators", min_value=10, max_value=300, value=100, step=1)
        loss = col1.selectbox("loss", ["log_loss", "exponential"], placeholder="Select loss", index=0)
        criterion = col1.selectbox("criterion", ["friedman_mse", "squared_error"], placeholder="Choose a criterion",
                                   index=0)
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features",
                                      index=0)
        ccp_alpha = col1.slider("ccp_alpha", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        n_jobs = col1.slider("slider", min_value=1, max_value=10, value=1, step=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        clf = get_gradient_boosting(
            criterion=criterion,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            random_state=random_state
        )


    elif ml_algorithm == "Random Forest Classifier":

        n_estimators = col1.slider("n_estimators", min_value=1, max_value=2000, value=100, step=1)
        criterion = col1.selectbox("criterion", ["gini", "entropy", "log_loss"], placeholder="Choose a criterion")
        max_depth = col1.slider("max_depth", min_value=3, max_value=200, value=3, step=1)
        min_samples_split = col1.slider("min_samples_split", min_value=2, max_value=100, value=2, step=1)
        min_samples_leaf = col1.slider("min_samples_leaf", min_value=1, max_value=200, value=1, step=1)
        min_weight_fraction_leaf = col1.slider("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0,
                                               step=0.1)
        max_features = col1.selectbox("max_features", ["sqrt", "log2", None], placeholder="Choose max. number of features")
        max_leaf_nodes = col1.slider("max_leaf_nodes", min_value=2, max_value=100, value=2, step=1)
        min_impurity_decrease = col1.slider("min_impurity_decrease", min_value=0.0, max_value=3.0, value=0.0, step=0.1)
        bootstrap = col1.checkbox("bootstrap", True)
        n_jobs = col1.slider("n_jobs", min_value=5, max_value=100, value=5, step=1)

        clf = get_random_forest(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            n_jobs=n_jobs
        )


    elif ml_algorithm == "One Class SVM":

        kernel = col1.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"], index=2)
        gamma = col1.selectbox("gamma", ["scale", "auto"], index=0)

        coef0 = col1.slider("coef0", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        cache_size = col1.slider("cache_size", min_value=100, max_value=1000, value=200, step=100)
        shrinking = col1.selectbox("shrinking", [True, False], index=0)


        clf = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            coef0=coef0,
            cache_size=cache_size,
            shrinking=shrinking
        )



    elif ml_algorithm == "Logistic Regression":

        penalty = col1.selectbox("penalty", ["l1", "l2", "elasticnet", None], index=1)
        dual = col1.selectbox("dual", [True, False], index=1)
        solver = col1.selectbox("solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        multi_class = col1.selectbox("multi_class", ["auto", "ovr", "multinomial"], index=0)

        clf = LogisticRegression(
            penalty=penalty,
            dual=dual,
            solver=solver,
            fit_intercept=fit_intercept,
            random_state=random_state,
            max_iter=max_iter,
            warm_start=warm_start,
            multi_class=multi_class
            )

    elif ml_algorithm == "Elastic Net":
        alpha = col1.slider("alpha", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
        l1_ratio = col1.slider("l1_ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=1000, step=100)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        positive = col1.selectbox("positive", [True, False], index=1)
        selection = col1.selectbox("selection", ['cyclic', 'random'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            copy_X=copy_X,
            warm_start=warm_start,
            positive=positive,
            selection=selection,
            random_state=random_state
        )


    elif ml_algorithm == "Gamma Regressor":
        alpha = col1.slider("alpha", min_value=1, max_value=100, value=1, step=1)
        solver = col1.selectbox("solver", ['lbfgs', 'newton-cholesky'], index=0)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = GammaRegressor(
            alpha=alpha,
            solver=solver,
            fit_intercept=fit_intercept,
            warm_start=warm_state,
            # random_state=random_state
        )
        
    elif ml_algorithm == "Gradient Boosting Regressor":
        loss = col1.selectbox("loss", ['squared_error', 'absolute_error', 'huber', 'quantile'], index=0)
        learning_rate = col1.slider("learning_rate", min_value=0.1, max_value=10.0, value=0.1, step=0.1)
        n_estimators = col1.slider("n_estimators", min_value=100, max_value=1000, value=100, step=10)
        criterion = col1.selectbox("criterion", ['friedman_mse', 'squared_error'], index=0)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)
        warm_state = col1.selectbox("warm_state", [True, False], index=1)

        regr = GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            random_state=random_state,
            warm_start=warm_state
        )


    elif ml_algorithm == "SGD Regressor":
        loss = col1.selectbox("loss", ["squared_error", "l1", "elasticnet", None], index=0)
        alpha = col1.slider("alpha", min_value=0.0001, max_value=1.000, value=0.0001, step=0.0001)
        fit_intercept = col1.selectbox("fit_intercept", [True, False], index=0)
        max_iter = col1.slider("max_iter", min_value=1000, max_value=10000, value=100, step=100)

        regr = SGDRegressor(
            fit_intercept=fit_intercept,
            loss=loss,
            alpha=alpha,
            max_iter=max_iter
        )
        

    elif ml_algorithm == "MLP Regressor":
        activation = col1.selectbox("activation", ['identity', 'logistic', 'tanh', 'relu'], index=3)
        solver = col1.selectbox("solver", ['lbfgs', 'sgd', 'adam'], index=2)
        learning_rate = col1.selectbox("learning_rate", ['constant', 'invscaling', 'adaptive'], index=0)
        max_iter = col1.slider("max_iter", min_value=100, max_value=5000, value=200, step=100)
        shuffle = col1.selectbox("shuffle", [True, False], index=0)
        warm_start = col1.selectbox("warm_start", [True, False], index=1)
        nesterovs_momentum = col1.selectbox("nesterovs_momentum", [True, False], index=0)
        early_stopping = col1.selectbox("early_stopping", [True, False], index=1)
        random_state = col1.slider("random_state", min_value=1, max_value=100, value=42, step=1)

        regr = MLPRegressor(
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            max_iter=max_iter,
            shuffle=shuffle,
            warm_start=warm_start,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            random_state=random_state
            )
        
    elif ml_algorithm == "Linear Regression":
        fit_intercept = col1.selectbox("fit_intercept", [True], index=0)
        copy_X = col1.selectbox("copy_X", [True, False], index=0)
        positive = col1.selectbox("positive", [True, False], index=1)

        regr = LinearRegression(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
        )
        
        
        
    if X is not None and y is not None and algo_type == "Classifiers":

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        model_buffer = io.BytesIO()
        joblib.dump(clf, model_buffer)
        model_buffer.seek(0)

        deployment_zip_buffer = create_zip(model_buffer=model_buffer)
        y_pred = clf.predict(X_test)

        col1.markdown("<center><h3>Metrics</h3></center>", unsafe_allow_html=True)
        col1.download_button(
            label="Download Model",
            data = model_buffer,
            file_name="model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )
        col1.download_button(
            label="Download Deployment Zip",
            data=deployment_zip_buffer,
            file_name="deployment.zip",
            mime="application/zip",
            use_container_width=True
        )
        col1.markdown(f"<b>Accuracy:</b> {accuracy_score(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Precision - Micro:</b> {precision_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Recall - Micro: </b> {recall_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>F1 Score - Micro: </b> {f1_score(y_test, y_pred, average='micro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Precision - Macro:</b> {precision_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Recall - Macro:</b> {recall_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>F1 Score - Macro:</b> {f1_score(y_test, y_pred, average='macro')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Precision - Weighted: </b> {precision_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Recall - Weighted:</b> {recall_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col1.markdown(f"<b>F1 Score - Weighted: </b> {f1_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        col1.markdown(f"<b>Classification Report:</b> {classification_report(y_test, y_pred)}", unsafe_allow_html=True)
        #col2.text(f"ROC AUC Score - OVR: {roc_auc_score(y_test, y_pred, multi_class='ovr')}")
        #col2.text(f"ROC AUC Score - OVO: {roc_auc_score(y_test, y_pred, multi_class='ovo')}")
        col1.markdown(f"<b>Confusion Matrix:</b> {confusion_matrix(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Hamming Loss: </b> {hamming_loss(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Jaccard Similarity Score: </b> {jaccard_score(y_test, y_pred, average='weighted')}", unsafe_allow_html=True)
        #col2.text(f"Log Loss: {log_loss(y_test, y_pred)}")
        col1.markdown(f"<b>Matthews Correlation Coefficient: </b> {matthews_corrcoef(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Balanced Accuracy:</b> {balanced_accuracy_score(y_test, y_pred)}", unsafe_allow_html=True)
        #col2.text(f"Precision-Recall Curve: {precision_recall_curve(y_test, y_pred)}")
        #col2.text(f"ROC Curve: {roc_curve(y_test, y_pred)}")
        col1.markdown(f"<b>Zero-One Loss:</b> {zero_one_loss(y_test, y_pred)}", unsafe_allow_html=True)
        st.toast("Yay!! Model prediction is complete!!", icon='🎉')
        time.sleep(.5)


    elif X is not None and y is not None and algo_type == "Regressors":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        regr.fit(X_train_scaled, y_train)
        model_buffer = io.BytesIO()
        joblib.dump(regr, model_buffer)
        model_buffer.seek(0)

        deployment_zip_buffer = create_zip(model_buffer=model_buffer)
        y_pred = regr.predict(X_test_scaled)

        col1.markdown("<center><h3>Metrics</h3></center>", unsafe_allow_html=True)
        col1.download_button(
            label="Download Model",
            data=model_buffer,
            file_name="model.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )

        col1.download_button(
            label="Download Deployment Zip",
            data=deployment_zip_buffer,
            file_name="deployment.zip",
            mime="application/zip",
            use_container_width=True
        )
        col1.markdown(f"<b>Mean Absolute Error:</b> {mean_absolute_error(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Mean Squared Error:</b> {mean_squared_error(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col1.markdown(f"<b>Mean Squared Error (Squared = False): </b> {mean_squared_error(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Mean Squared Log Error: </b> {mean_squared_log_error(y_test, y_pred)}", unsafe_allow_html=True)
        col1.markdown(f"<b>Median Absolute Error:</b> {median_absolute_error(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col1.markdown(f"<b>R2 Score:</b> {r2_score(y_test, y_pred)}",
                      unsafe_allow_html=True)
        col1.markdown(f"<b>Explained Variance Score:</b> {explained_variance_score(y_test, y_pred)}",
                      unsafe_allow_html=True)


if __name__ == "__main__":
    main()

