from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
        median_income = float(request.form["median_income"])
        
        # Make prediction
        prediction = model.predict(np.array([[median_income]]))[0]
        
        return render_template("index.html", prediction=round(prediction, 2))
    
    except Exception as e:
        return render_template("index.html", error="Invalid input! Please enter a valid number.")

if __name__ == "__main__":
    app.run(debug=True)
