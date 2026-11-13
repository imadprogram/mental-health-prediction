from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import pandas as pd



filepath = '../data/student_mental_health.csv'





def load_and_split_data(filepath):
    
    data = pd.read_csv(filepath)

    data = data.drop(columns=['Student_ID'])

    X = data.drop("Mental_Health_Risk", axis=1)
    y = data['Mental_Health_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    y_train = y_train.map(risk_map)
    y_test = y_test.map(risk_map)

    return X_train, X_test, y_train, y_test







def build_preprocessor(X_train):
    
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    
    
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    num_pipeline = Pipeline([
        ('imputer', num_imputer),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', cat_imputer),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    
    return preprocessor



