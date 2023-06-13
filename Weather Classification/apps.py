import numpy as np 
import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

model=load_model("wcv.h5")
app=Flask (__name__, template_folder='templates/')
app.config['UPLOAD_FOLDER'] = 'uploads/'


@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/images')
def image():
    return render_template("images.html")


@app.route('/input')
def input():
    return render_template("predict.html")

@app.route('/predict', methods=["GET", "POST"])
def res():
    if request.method=="POST":
        f=request.files['image'] 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)
        img = Image.open(filepath)
        img = img.resize((180, 180))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        preds=model.predict(x)
        pred=np.argmax(preds,axis=1)
        index=['cloudy','foggy','rainy','shine','sunrise']
        result=str(index[pred[0]])
        return render_template('result.html', prediction=result, path = f.filename)

if __name__==  "__main__":
    app.run(debug=True)






