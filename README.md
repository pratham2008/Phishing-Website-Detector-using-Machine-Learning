🛡️ Phishing Website Detector using Machine Learning

A user-friendly web application built with Streamlit that leverages a Random Forest classifier to detect phishing websites in real-time. This project was developed to provide a simple yet effective tool to combat the growing threat of phishing attacks by analyzing various URL features.


✨ Features
Real-Time Analysis: Instantly analyze URLs by pasting them into the input field.

Interactive UI: A clean and intuitive user interface built with Streamlit for ease of use.

Machine Learning Backend: Powered by a Scikit-learn Random Forest model trained on a dataset of over 11,000 URLs.

Feature-Based Detection: Analyzes 30 distinct features of a URL, including its structure, domain properties, and character patterns, to make accurate predictions.

🔧 Technology Stack
Backend: Python

Machine Learning: Scikit-learn, Pandas

Web Framework: Streamlit

Data Handling: Joblib, NumPy

🚀 Setup and Installation
Follow these steps to set up and run the project on your local machine.

Prerequisites
Python 3.8 or higher

pip (Python package installer)

1. Clone the Repository
First, clone the project from your GitHub repository to your local machine.

# Replace 'your-username' with your actual GitHub username
git clone [https://github.com/your-username/phishingdetection.git]([https://github.com/your-username/phishingdetection.git](https://github.com/pratham2008/Phishing-Website-Detector-using-Machine-Learning.git))
cd phishingdetection

2. Install Dependencies
It's highly recommended to create a virtual environment to keep the project's dependencies isolated.

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Now, install all the required libraries using the requirements.txt file.

pip install -r requirements.txt

3. Train the Machine Learning Model
Before you can run the application, you need to train the model. This script will read the dataset.csv file and create the phishing_model.pkl file that the web app needs.

python train.py

4. Run the Streamlit Application
You are now ready to launch the web application!

streamlit run app.py

If the command above doesn't work, you can also use:

python -m streamlit run app.py

Streamlit will start the server and your browser should automatically open a new tab with the application running.

📋 How to Use
Once the application is running, you will see the main interface.

Enter a full website URL into the text box (e.g., https://www.google.com).

Click the "Analyze URL" button.

The application will process the URL and display the prediction result: Legitimate or Phishing.

⚙️ How It Works
The project's detection capability is based on a two-step process:

Feature Extraction: When a user enters a URL, the application extracts 30 specific features from it. These features are based on common phishing patterns, such as the use of an IP address, abnormal URL length, use of shortening services, presence of special characters like "@" and "-", and the number of subdomains.

Prediction: The extracted feature vector (a NumPy array of shape [1, 30]) is then fed into the pre-trained Random Forest model (phishing_model.pkl). The model, having learned from thousands of examples, classifies the vector and returns a prediction: -1 for a legitimate site and 1 for a phishing site.
