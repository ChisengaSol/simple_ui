from flask import Flask, render_template,request
import pickle

tokenizer = pickle.load(open("models/cv.pkl","rb"))
model = pickle.load(open("models/clf.pkl","rb"))

#create an instance of the flask class
app = Flask(__name__)

#create a route
@app.route("/") #this serves as a decorator
def home():
    #you render the template for it to appear in browser
    #render_template looks in the templates folder so we dont have to specify templates/*
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        email_text = request.form.get("email-content")

    #we now tokenize the text
    tokenize_email = tokenizer.transform([email_text])
    predictions = model.predict(tokenize_email)

    predictions = 1 if predictions == 1 else -1
    return render_template('index.html',predictions=predictions, email_text=email_text)

#run the flask app
if __name__ == '__main__':
    app.run(debug=True) #debug = True ensures that when you make a change it shows up