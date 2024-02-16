from flask import Flask, request, render_template, redirect, url_for, send_file
from scannergit import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('loading.html')

@app.route('/result', methods=['GET'])
def result():
    username = request.args.get('username')
    extension = request.args.get('extension')
    
    extension = tuple(extension.split(" "))
    
    list_folders_for_user(username, "YOUR_TOKEN_HERE", extension)
    
    filepath = "data/" + username + ".txt"
    
    return send_file(filepath, as_attachment=True)

    

if __name__ == '__main__':
    app.run(debug=True)