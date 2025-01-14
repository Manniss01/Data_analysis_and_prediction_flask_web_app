import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from io import BytesIO
import base64
import joblib

app = Flask(__name__)

# Configuring the upload folder for CSV files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16 MB

# Load the pre-trained house price model
def load_model():
    return joblib.load('house_price_model.pkl')

model = load_model()

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the house price prediction
@app.route('/predict', methods=['POST'])
def predict():
    size = float(request.form['size'])
    bedrooms = int(request.form['bedrooms'])

    prediction = model.predict([[size, bedrooms]])
    prediction_text = f"The predicted house price is: ${prediction[0]:,.2f}"

    return render_template('index.html', prediction_text=prediction_text)

# Route for handling the CSV file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the file to the uploads folder
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform analysis and generate a plot
        data = pd.read_csv(filename)

        # Log and display the column names to the console for debugging
        column_names = data.columns.tolist()
        print("Uploaded CSV Columns:", column_names)

        # Show the column names on the page for debugging
        column_names_str = ', '.join(column_names)

        # Automatically identify numeric columns
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_columns) < 2:
            return render_template('index.html', error_message="Not enough numeric columns to generate analysis. Upload a valid file with numerical data.", column_names=column_names_str)

        # Assume that the first two numeric columns are the ones we need for the plot (adjust logic as necessary)
        size_column = numeric_columns[0]
        price_column = numeric_columns[1]

        # Generate the plot
        plot_urls = generate_plots(data, size_column, price_column)
        return render_template('index.html', plot_urls=plot_urls, data=data.to_html(classes='data'), column_names=column_names_str)

    return redirect(url_for('index'))

# Function to generate plots (scatter, bar, and pie charts)
def generate_plots(data, size_column, price_column):
    # Prepare a dictionary to store the plot URLs
    plot_urls = {}

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.scatterplot(x=size_column, y=price_column, data=data, color='blue', s=100)
    plt.title(f'{price_column.capitalize()} vs {size_column.capitalize()}')
    scatter_img = BytesIO()
    plt.savefig(scatter_img, format='png')
    scatter_img.seek(0)
    plot_urls['scatter'] = base64.b64encode(scatter_img.getvalue()).decode('utf8')
    plt.close()

    # Bar Chart (example: showing the count of houses per price range or category)
    price_bins = pd.cut(data[price_column], bins=10)  # Bin prices into 10 categories
    bar_plot = data.groupby(price_bins).size().plot(kind='bar', figsize=(10, 6), color='green')
    plt.title(f'{price_column.capitalize()} Distribution')
    bar_img = BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    plot_urls['bar'] = base64.b64encode(bar_img.getvalue()).decode('utf8')
    plt.close()

    # Pie Chart (example: showing the distribution of 'Bedrooms' or any other categorical column)
    if 'Bedrooms' in data.columns:
        bedroom_counts = data['Bedrooms'].value_counts()
        pie_plot = bedroom_counts.plot(kind='pie', figsize=(8, 8), autopct='%1.1f%%', colors=sns.color_palette("Set3", len(bedroom_counts)))
        plt.title('Distribution of Bedrooms')
        pie_img = BytesIO()
        plt.savefig(pie_img, format='png')
        pie_img.seek(0)
        plot_urls['pie'] = base64.b64encode(pie_img.getvalue()).decode('utf8')
        plt.close()

    return plot_urls

if __name__ == '__main__':
    app.run(debug=True)
