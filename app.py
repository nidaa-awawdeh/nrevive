
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import spacy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('coma_patient_notes.csv')

# Preprocess the data
X = dataset['medical_notes']
y_severity = dataset['coma_severity']
y_recovery = dataset['recovery_likelihood']
y_complication = dataset['complication_risk']

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X_vect = vectorizer.fit_transform(X)

# Train machine learning models
model_severity = LogisticRegression()
model_severity.fit(X_vect, y_severity)

model_recovery = RandomForestClassifier()
model_recovery.fit(X_vect, y_recovery)

model_complication = RandomForestClassifier()
model_complication.fit(X_vect, y_complication)

# Initialize and train Decision Trees model
model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(X_vect, y_severity)

# Load NLP model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Save trained models
joblib.dump(vectorizer, 'vectorizer.pkl', protocol=4)
joblib.dump(model_severity, 'model_severity.pkl', protocol=4)
joblib.dump(model_recovery, 'model_recovery.pkl', protocol=4)
joblib.dump(model_complication, 'model_complication.pkl', protocol=4)
joblib.dump(model_decision_tree, 'model_decision_tree.pkl', protocol=4)
joblib.dump(nlp, 'nlp_model.pkl', protocol=4)

# Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/about.html')
def about():
    return render_template('about.html')



@app.route('/static/<path:path>')
def static_file(path):
    return app.send_static_file(path)

@app.route('/model.html')
def model():
    return render_template('model.html')
@app.route('/result.html')
def casestudy():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            input_text = request.form.get('medical_notes')
            
            if input_text is None or input_text.strip() == '':
                return render_template('model.html', error_message="Please enter medical notes.")
            
            # Load models
            vectorizer = joblib.load('vectorizer.pkl')
            model_severity = joblib.load('model_severity.pkl')
            model_recovery = joblib.load('model_recovery.pkl')
            model_complication = joblib.load('model_complication.pkl')
            model_decision_tree = joblib.load('model_decision_tree.pkl')
            nlp = joblib.load('nlp_model.pkl', mmap_mode=None)
            
            # Vectorize the input text
            input_text_vect = vectorizer.transform([input_text])
            
            # Make predictions
            prediction_severity = model_severity.predict(input_text_vect)
            prediction_recovery = model_recovery.predict(input_text_vect)
            prediction_complication = model_complication.predict(input_text_vect)
            prediction_decision_tree = model_decision_tree.predict(input_text_vect)
            
            # NLP tasks
            doc = nlp(input_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [(token.text, token.pos_) for token in doc]
            
            return render_template('model.html', 
                                   prediction_severity=prediction_severity,
                                   prediction_recovery=prediction_recovery,
                                   prediction_complication=prediction_complication,
                                   prediction_decision_tree=prediction_decision_tree,
                                   entities=entities,
                                   pos_tags=pos_tags)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('model.html', error_message=error_message)



##case study

# Load data
try:
    df = pd.read_csv('laboratory_findings.csv')
except FileNotFoundError:
    df = None

# Define model outside the route functions
model = None

# Train model if data is available
if df is not None:
    try:
        X = df.drop('Diagnosis', axis=1)
        y = df['Diagnosis']
    except KeyError:
        X = df.copy()
        y = None

    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

# Define routes
@app.route('/casestudy')
def casestudy_index():
    return render_template('result.html')

@app.route('/casestudypredict', methods=['POST'])
def casestudy_predict():
    global model  # Ensure you're referencing the global model object
    if request.method == 'POST':
        if X is not None and model is not None:
            data = request.form.to_dict()
            input_data = pd.DataFrame.from_dict(data, orient='index').T
            prediction = model.predict(input_data)
            return prediction[0]  # Return prediction result as plain text
        else:
            return "Model not trained or data not available"
if __name__ == '__main__':
    app.run(debug=True)

