import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, mean_squared_error
from streamlit_extras.metric_cards import style_metric_cards

st.markdown(
    """
    <style>
        /* Use CSS variables for easy palette adjustments */
        :root {
            --primary-color: #ff4b4b;
            --primary-hover: #e04343;
            --secondary-color: #212529;
            --background-gradient: linear-gradient(135deg, #1f1f1f, #262626);
            --font-family: 'Roboto', sans-serif;
            --transition-speed: 0.3s;
            --heading-color: #ff4b4b;
            --text-color: #f1f1f1;
            --sidebar-text-color: #f8f9fa;
        }

        /* Smooth fade-in for the entire app */
        @keyframes fadeIn {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        .stApp {
            background: var(--background-gradient);
            color: var(--text-color);
            font-family: var(--font-family);
            animation: fadeIn 1s ease-in;
        }
        /* Keep the main container transparent so the gradient shows through */
        .main {
            background: transparent;
        }
        /* Dark sidebar styling */
        [data-testid="stSidebar"] {
            background: var(--secondary-color);
            color: var(--sidebar-text-color);
            font-family: var(--font-family);
        }
        /* Sidebar headers in a warm accent color */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffc107;
        }
        /* Main headers */
        h1, h2, h3 {
            color: var(--heading-color);
            font-family: var(--font-family);
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }
        /* Buttons: smooth hover scale */
        div.stButton > button:first-child {
            background-color: var(--primary-color);
            color: #fff;
            border-radius: 10px;
            font-size: 18px;
            border: none;
            transition:
                background-color var(--transition-speed) ease,
                transform var(--transition-speed) ease;
        }
        div.stButton > button:first-child:hover {
            background-color: var(--primary-hover);
            transform: scale(1.05);
        }
        /* Inputs: subtle glow on hover */
        div.stSelectbox, div.stRadio, .stTextInput {
            font-size: 16px;
            border: 1px solid #555;
            border-radius: 5px;
            background-color: #333;
            color: #fff;
            padding: 0.5rem;
            transition: box-shadow var(--transition-speed) ease;
        }
        div.stSelectbox:hover, div.stRadio:hover, .stTextInput:hover {
            box-shadow: 0 0 8px rgba(255,255,255,0.2);
        }
        /* Optional: darker background for tables / dataframes */
        .css-1aumxhk, .css-1xarl3l, .css-2ogi0 {
            background-color: #2c2c2c !important;
            color: #f1f1f1 !important;
        }
        /* Metric cards: slight lift on hover */
        .metric-card {
            background-color: #333333;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
            padding: 1rem;
            margin: 0.5rem;
            transition: transform var(--transition-speed) ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        /* Footer styling (fixed at bottom) */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: var(--secondary-color);
            color: #fff;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            opacity: 0.9;
            z-index: 100;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Your main Streamlit code...
st.title("📈 Telco Customer Analysis Dashboard")
st.write("Compare supervised learning models for customer churn prediction (classification) and revenue prediction (regression).")

# ============================
# 🚀 Load Data
# ============================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("Telco Customer Churn Cleaned.csv")
st.write("### 📊 Data Preview", df.head())

# Features & Targets
X = df.drop(['Churn', 'TotalCharges'], axis=1)
y_class = df['Churn']
y_reg = df['TotalCharges']

# Train-Test Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.3, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)

# ============================
# 🎯 Sidebar - Task Selection
# ============================
st.sidebar.header("🛠 Select Task")
task = st.sidebar.radio("Choose a task", ("Classification", "Regression"))



# ============================
# 🏆 Classification Section
# ============================
if task == "Classification":
    st.header("🔍 Customer Churn Prediction")

    st.sidebar.header("⚙️ Classification Options")
    classifier_name = st.sidebar.selectbox(
        "Select Classifier",
        ("Logistic Regression", "Support Vector Machine", "Naive Bayes", "Decision Tree", "Random Forest", "k-Nearest Neighbors")
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Classifier Selection
    if classifier_name == "Logistic Regression":
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=200)
    elif classifier_name == "Support Vector Machine":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        kernel = st.sidebar.radio("Kernel", ("linear", "rbf"))
        model = SVC(C=C, kernel=kernel, probability=True)
    elif classifier_name == "Naive Bayes":
        model = GaussianNB()
    elif classifier_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif classifier_name == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100)
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif classifier_name == "k-Nearest Neighbors":
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Model Training
    if st.sidebar.button("🚀 Train Classification Model"):
        model.fit(X_train_clf, y_train_clf)
        accuracy = model.score(X_test_clf, y_test_clf)

        st.metric("📊 Accuracy", f"{accuracy:.2f}")

        y_pred = model.predict(X_test_clf)

        # Confusion Matrix
        cm = confusion_matrix(y_test_clf, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

        # ROC Curve
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_clf)[:, 1]
        else:
            y_scores = model.decision_function(X_test_clf)
        fpr, tpr, _ = roc_curve(y_test_clf, y_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC curve")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        st.pyplot(fig)

# ============================
# 💰 Regression Section
# ============================
elif task == "Regression":
    st.header("💰 Revenue Prediction")

    st.sidebar.header("⚙️ Regression Options")
    regressor_name = st.sidebar.selectbox(
        "Select Regressor",
        ("Linear Regression", "Ridge Regression", "Lasso Regression", "Support Vector Regressor", "Decision Tree Regressor", "Random Forest Regressor", "k-Nearest Neighbors Regressor")
    )

    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor

    # Regressor Selection
    if regressor_name == "Linear Regression":
        model = LinearRegression()
    elif regressor_name == "Ridge Regression":
        alpha = st.sidebar.slider("Alpha", 0.1, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    elif regressor_name == "Lasso Regression":
        alpha = st.sidebar.slider("Alpha", 0.1, 10.0, 1.0)
        model = Lasso(alpha=alpha)
    elif regressor_name == "Support Vector Regressor":
        C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        model = SVR(C=C)
    elif regressor_name == "Decision Tree Regressor":
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif regressor_name == "Random Forest Regressor":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100)
        model = RandomForestRegressor(n_estimators=n_estimators)
    elif regressor_name == "k-Nearest Neighbors Regressor":
        n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    if st.sidebar.button("🚀 Train Regression Model"):
        # Drop NaNs in y_train_reg and keep corresponding X_train_reg rows
        mask = ~y_train_reg.isna()
        X_train_reg = X_train_reg[mask]
        y_train_reg = y_train_reg[mask]

        # Train the model
        model.fit(X_train_reg, y_train_reg)

        # Fill NaNs in test data
        X_test_reg.fillna(X_test_reg.mean(), inplace=True)
        y_test_reg.fillna(y_test_reg.mean(), inplace=True)

        # Get R² Score
        r2 = model.score(X_test_reg, y_test_reg)
        st.metric("📊 R² Score", f"{r2:.2f}")

st.markdown(
    """
    <div class="footer">
        © 2025 Your Company Name - All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)