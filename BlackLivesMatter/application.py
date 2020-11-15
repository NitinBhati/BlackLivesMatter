import os
import webbrowser
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
#//commented line 138
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,make_response,Response
from flask import send_file
from flask import send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import csv
# import numpy
import compas
import argparse
import math
import numpy as np
# import pandas as pd
import pylab
import fairness
import fairness_plotter
from flask import Blueprint
import jinja2
from PyPDF2 import PdfFileWriter, PdfFileReader
# from flask import Flask, render_template, request
# from flaskr import application
application = Flask(__name__)
import flask

# #********************* main code ****************
# This is the path to the upload directory
application.config['UPLOAD_FOLDER'] = 'C:/Users/Nitin_Bhati/fairness-Compas/flaskr/uploads'
    # These are the extension that we are accepting to be uploaded
application.config['ALLOWED_EXTENSIONS'] = set(['csv'])
global filename
global sensitive_attr_col
global predicted_outcome
global direct_attribute
global true_outcome
# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in application.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload_page')
def upload_page():
    return render_template('upload_page.html')

@application.route('/about_page')
def about_page():
    return render_template('about_page.html')

@application.route('/contact_page')
def contact_page():
    return render_template('contact_page.html')

@application.route('/intro_page')
def intro_page():
    return render_template('intro_page.html')

import pyexcel as pe
# global filename
# Route that will process the file upload
@application.route('/upload', methods=['POST'])
def upload():
    global filename
    global sensitive_attr_col
    global true_outcome
    global predicted_outcome
    global direct_attribute
    # Get the name of the uploaded file
    file = request.files['file']
    sensitive_attr_col = request.form['sensitive_attr']
    true_outcome = request.form['outcome_attr']
    predicted_outcome = request.form['pred_outcome']
    direct_attribute = request.form['direct_attr']
    print sensitive_attr_col
    print true_outcome
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars

        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('result'))

# @application.route('/uploads/<filename>')
@application.route('/uploaded_file/')
# @application.route('/display/filename')
# @application.route('/display.html')
def uploaded_file():

    dir = "C:/Users/Nitin_Bhati/fairness-Compas/flaskr/uploads/"
    # # with open(os.path.join(dir+filename)) as f_1:
    # env = jinja.Environment()
    # env.loader = jinja.FileSystemLoader("C:/Users/Nitin_Bhati/fairness-Compas/flaskr/flaskr/templates")
    # template = env.get_template("display.html")

    with open('C:/Users/Nitin_Bhati/fairness-Compas/flaskr/uploads/'+str(filename),'rb') as csvFile:
        my_file = csv.reader(csvFile,delimiter=',')
        # csv_data = [row for row in my_file]
        # print template.render(data = csv_data)
        a = []
        # i=0
        for row in my_file:
        # for i in xrange(0,10):
            # i=i+1
            # return (str(i))
            a.append(row)
            #a.applicationend(row)
        a= pd.DataFrame(a)
        # "<b>Hello World!</b>"
        # return render_template("display.html",my_file= my_file)
        return str(a.to_html())
#
# # # **************testing************************
# @application.route('/')
# def index():
#     return render_template('test.html')
# @application.route('/run/')
@application.route('/result')
def result():
    # return "Hello World Run!"
    execfile('compas.py')
    # return "Hello World!"  //commented below line 
    #print ("Hello World"+str(compas.main(filename,sensitive_attr_col,true_outcome,predicted_outcome,direct_attribute)))
    with open("C:/Users/Nitin_Bhati/fairness-Compas/flaskr/result2.txt","r") as f:
        content = f.read()
    return render_template("result.html", content= content)
#     # return flask.Response(compas.main(), mimetype='text/html')  # text/html is required for most browsers to show the partial page immediately

@application.route('/target-pdf/')
def threshold_results():
    req_pdf = "compas-thresholds.pdf"
    # response.headers['Content-Disposition'] = 'inline; filename= compas-thresholds.pdf'
    return send_from_directory(directory="C:/Users/Nitin_Bhati/fairness-Compas/flaskr/figs/",filename=req_pdf,mimetype="applicationlication/pdf", as_attachment=False)
    # print file_content_var
    # return response


@application.route('/maxprofit/')
def maxprofit():
    req_pdf = "compas-fixed.pdf"
    return send_from_directory(directory="C:/Users/Nitin_Bhati/fairness-Compas/flaskr/figs/", filename=req_pdf, mimetype="applicationlication/pdf",
                               as_attachment=False)

    # return render_template("maxprofit_results.html")

@application.route('/profit/')
def profit():
    req_pdf = "compas-targets.pdf"
    return send_from_directory(directory="C:/Users/Nitin_Bhati/fairness-Compas/flaskr/figs/", filename=req_pdf, mimetype="applicationlication/pdf",
                               as_attachment=False)

    # return render_template("profit_results.html")

@application.route('/roc/')
def roc():
    req_pdf = "compas-roc.pdf"
    return send_from_directory(directory="C:/Users/Nitin_Bhati/fairness-Compas/flaskr/figs/", filename=req_pdf, mimetype="applicationlication/pdf",
                               as_attachment=False)

    # return render_template("roc_resutls.html")

@application.route('/performance/')
def performance():
    req_pdf = "compas-marginals.pdf"
    return send_from_directory(directory="C:/Users/Nitin_Bhati/fairness-Compas/flaskr/figs/", filename=req_pdf, mimetype="applicationlication/pdf",
                               as_attachment=False)

    # return render_template("performance_results.html")

if __name__ == '__main__':
    application.run(
        host="127.0.0.1",
        port=int("5000"),
        debug=True
    )
