from flask import Flask,render_template,flash,redirect,request,send_from_directory,url_for, send_file
import mysql.connector, os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import seaborn as sns

app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='fish'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('index.html', message="Successfully Registered! Please go to login section")
            return render_template('index.html', message="This email ID is already exists!")
        return render_template('index.html', message="Conform password is not match!")
    return render_template('index.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('index.html', message= "Invalid Password!!")
        return render_template('index.html', message= "This email ID does not exist!")
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


'''
def fish_demand_info(fish_name, df):
    fish_data = df[df['Fish Type'] == fish_name]
    if fish_data.empty:
        return f"No information available for {fish_name}"
    
    avg_consumption = fish_data['Average Consumption (tons)'].mean()
    preferred_size = fish_data['Preferred Size (cm)'].mean()
    price_low = fish_data['Price Range (INR per kg)'].apply(lambda x: float(x.split(' - ')[0])).mean()
    price_high = fish_data['Price Range (INR per kg)'].apply(lambda x: float(x.split(' - ')[1])).mean()
    seasonal_availability = fish_data['Seasonal Availability'].mode().iloc[0]

    info = f"Demand info for {fish_name}:\n"
    info += f"Average consumption: {avg_consumption:.2f} tons\n"
    info += f"Preferred size: {preferred_size:.2f} cm\n"
    info += f"Price range: INR {price_low:.2f} - {price_high:.2f} per kg\n"
    info += f"Seasonal availability: {seasonal_availability}"

    plt.figure(figsize=(10, 6))
    plt.bar(['Average Consumption', 'Preferred Size', 'Price Range Low', 'Price Range High'],
            [avg_consumption, preferred_size, price_low, price_high],
            color=['blue', 'green', 'orange', 'red'])
    plt.title(f'Demand Info for {fish_name}')
    plt.ylabel('Values')
    plt.xlabel('Statistics')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    # img_tag = f'<img src="data:image/png;base64,{img_str}"/>'
    plt.close()
    return img_str

'''

import matplotlib.pyplot as plt

def plot_fish_demand(fish_name, df):
    fish_data = df[df["Fish Type"] == fish_name]
    purchases = fish_data["Purchases"].apply(lambda x: int(x.split()[0]))
    plt.figure(figsize=(10, 6))
    plt.plot(purchases.reset_index(drop=True), marker='o', linestyle='-', color='blue')
    plt.title(f'Demand for {fish_name}', fontsize=15)
    plt.xlabel('Data Point', fontsize=12)
    plt.ylabel('Purchases (Units)', fontsize=12)
    plt.grid(True)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{img_str}"/>'
    plt.close()
    return img_tag


import numpy as np  # Add this import for numerical operations

def plot_fish_demand_detailed(fish_name, df):
    fish_data = df[df["Fish Type"] == fish_name]
    seasons = fish_data["Seasonal Data"].unique()
    n_seasons = len(seasons)
    # Assuming 'Purchases' column values are strings and need to be converted to integers
    fish_data["Purchases"] = fish_data["Purchases"].apply(lambda x: int(x.split()[0]))

    plt.figure(figsize=(12, 8))

    # Width of each bar and the spacing between groups of bars
    bar_width = 0.8 / n_seasons  # Adjust this value as needed
    index = np.arange(len(fish_data) / n_seasons)  # Create a set of index values for the x-axis
    
    for i, season in enumerate(seasons):
        season_data = fish_data[fish_data["Seasonal Data"] == season]
        # Create an offset for each season to position bars side by side
        offset = (i - n_seasons / 2) * bar_width + bar_width / 2
        plt.bar(index + offset, season_data["Purchases"], bar_width, label=season)

    plt.title(f'Detailed Demand Analysis for {fish_name}', fontsize=15)
    plt.xlabel('Data Point', fontsize=12)
    plt.ylabel('Purchases (Units)', fontsize=12)
    plt.xticks(index, rotation=45)  # Adjust the rotation if labels overlap
    plt.legend(title="Season")
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{img_str}"/>'
    plt.close()
    return img_tag

def plot_fish_demand_detailed(fish_name, df):
    fish_data = df[df["Fish Type"] == fish_name]
    purchases = fish_data["Purchases"].apply(lambda x: int(x.split()[0]))
    seasons = fish_data["Seasonal Data"]
    plt.figure(figsize=(12, 8))
    for season in seasons.unique():
        season_data = fish_data[fish_data["Seasonal Data"] == season]
        season_purchases = season_data["Purchases"].apply(lambda x: int(x.split()[0]))
        plt.plot(season_purchases.reset_index(drop=True), marker='o', linestyle='-', label=season)
    plt.title(f'Detailed Demand Analysis for {fish_name}', fontsize=15)
    plt.xlabel('Data Point', fontsize=12)
    plt.ylabel('Purchases (Units)', fontsize=12)
    plt.legend(title="Season")
    plt.grid(True)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{img_str}"/>'
    plt.close()
    return img_tag


@app.route('/classification', methods=["GET", "POST"])
def classification():
    if request.method == "POST":
        select = request.form["select"]
        if select == "0":
            return render_template("classification.html", message = "Can not predict a non fish Image! Please give a fish image")
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('static/img/', fn)
        myfile.save(mypath)
        accepted_formated=['jpg','png','jpeg','jfif','JPG']
        if fn.split('.')[-1] not in accepted_formated:
             return render_template("classification.html", message = "This image format can not be accept!")
        
        classes=["Black Sea Sprat","Gilt-Head Bream","Hourse Mackerel","Red Mullet","Red Sea Bream","Sea Bass","Shrimp","Striped Red Mullet","Trout"]
        new_model = load_model("resnet_1.h5")
        test_image = image.load_img(mypath, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image/255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        fish_name=classes[np.argmax(result)]

        df=pd.read_csv("fish_analysis.csv")
        chart_html1 = plot_fish_demand_detailed(fish_name, df)
        chart_html2 = plot_fish_demand(fish_name, df)
        return render_template('classification.html', fish = fish_name, path = mypath, chart_html1 = chart_html1, chart_html2 = chart_html2)
    return render_template('classification.html')


@app.route("/chart", methods=["POST"])
def chart():
    if request.method == "POST":
        fish_name = request.form['fish']
        chart_html1 = request.form['chart_html1']
        chart_html2 = request.form['chart_html2']

        return render_template('result.html', fish = fish_name, chart_html1 = chart_html1, chart_html2 = chart_html2) 



@app.route('/recommentation', methods=["GET", "POST"])
def recommentation():
    if request.method == "POST":
        def fish_recommendations_for_state(state, df):
            user_item_matrix = df.pivot_table(index='State', columns='Fish Type', values='Average Consumption (tons)', fill_value=0)
            sparse_matrix = csr_matrix(user_item_matrix.values)
            k = min(user_item_matrix.shape) - 1  # Number of latent factors (choose one less than the smaller dimension)
            U, sigma, Vt = svds(sparse_matrix, k=k)
            predicted_consumption = np.dot(np.dot(U, np.diag(sigma)), Vt)
            predicted_df = pd.DataFrame(predicted_consumption, columns=user_item_matrix.columns, index=user_item_matrix.index)
            state_consumption = predicted_df.loc[state]
            recommended_fish = state_consumption.sort_values(ascending=False).head(3)
            return recommended_fish.index.tolist()
        
        df=pd.read_csv("fish_demand_dataset_india_2000.csv")
        state = request.form['state']
        result = fish_recommendations_for_state(state, df)
        return render_template('recommentation.html', result=result)
    return render_template('recommentation.html')




if __name__ == '__main__':
    app.run(debug=True, threaded=False)
