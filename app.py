from flask import Flask, render_template, request,redirect,url_for
import jsonify
import requests
import urllib.request
import pickle
import os
import numpy as np
import sklearn
from codesim import qasystem    



app = Flask(__name__)


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST':
        question=request.form["question"]
        option=request.form["word"]
        q1=qasystem()
        if (option=="bowe"):
                rques,ans=q1.bow(question)
                return render_template('answer.html', question=question,rquestion=rques,answer=ans)
        elif (option=="tfide"):       
            rques,ans=q1.tfidf(question)
            return render_template('answer.html', question=question,rquestion=rques,answer=ans)
        elif (option=="wtve"):
            rques,ans=q1.w2vec(question)
            return render_template('answer.html', question=question,rquestion=rques,answer=ans) 
        elif (option=="gloe"):
            rques,ans=q1.glove(question)
            return render_template('answer.html', question=question,rquestion=rques,answer=ans)
       
    else:
        return render_template('index.html')

@app.route('/refresh', methods = ['POST'])
def refresh():
    return render_template('index.html')



@app.route('/back', methods = ['POST'])
def back():
    return redirect("/")


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__=="__main__":
    app.run(debug=True)