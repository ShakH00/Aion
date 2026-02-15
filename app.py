from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from pathlib import Path
import users
from users import user
from pymongo import MongoClient
import os

client = MongoClient(os.getenv("DB_KEY"))
db = client['aionDB']
user_c = db['users']
app = Flask(__name__)
INDEX_DIR = Path(app.root_path) / "index"


@app.route("/")
def index():
    return send_from_directory(INDEX_DIR, "index.html")


#helper method to get the username (first and last name) of an account
def getUserName():
    all_users = users.get_all_users()
    userName = ""
    for user in all_users:
        if user['email'] == session['email']:
            userName += f"{user['name']} {user['lastname']}"
    return userName


#login method
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('email')
        pwd = request.form.get('password')
        email, password = users.get_user_credentials(username)
        #verify password
        if user.verify_password(password, pwd):  # Simulate successful login
            session['email'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid email or password!')
    return render_template('login.html')


#register account
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        pwd = request.form.get('password')
        first = request.form.get('first')
        last = request.form.get('last')

        if email is None or pwd is None or first is None or last is None:
            return render_template('register.html', error="Missing information!")

        exists = False
        all_users = users.get_all_users()
        #for loop and if statement used to make sure the email isn't already in use
        for usr in all_users:
            if usr['email'] == email:
                exists = True
        if exists == True:
            return render_template('register.html', error="Email already in use!")
        else:
            new_person = user(first, last, email, pwd)
            new_person.insert_doc()

            # Simulate user registration
            session['email'] = email  # Log the user in after registration
            return redirect(url_for('login'))
    return render_template('register.html')


#upload pdf
@app.route('/uploadpdf', methods=['GET', 'POST'])
def uploadpdf():
    return render_template('uploadpdf.html') #temp

#logout method
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
	app.run(debug=True)
