# Real-time-call-classification-system

**Creat a 2 text(.py) documents, 1st one being p1.py**

**Step 1: Install Necessary Libraries**
In the command prompt, install the required libraries:

_pip install pandas numpy scikit-learn nltk tensorflow keras flask_

These libraries will support data handling, machine learning, NLP preprocessing, and deployment.

![image](https://github.com/user-attachments/assets/1b8fd342-ae9b-47ce-bd4d-6b0b9d60f526)

**Step 2: Import Libraries and Load Data**
Import the libraries we need in the p1.py. We'll then define a sample dataset of labeled conversations to train our model:

![image](https://github.com/user-attachments/assets/727d5570-ba8f-4f33-9a79-fbc6a5d25424)

**Step 3: Preprocess the Data**
Data Cleaning and Preprocessing: Tokenize text, remove stopwords, and convert to lower case for uniformity.
Text Vectorization: Use TfidfVectorizer to convert text data into numerical features for the model.

![image](https://github.com/user-attachments/assets/87c6a0c4-8417-4c5c-b363-f2cb697980ac)

**Step 4: Set Up Training Pipeline**
Used a simple model (Logistic Regression) in a pipeline with TF-IDF vectorization. For a large dataset, consider using deep learning (e.g., LSTM with Keras).

![image](https://github.com/user-attachments/assets/c303be1d-4cac-4ce0-b9e1-1d950a8a6621)

**Step 5: Set Up Real-Time Classification**
For real-time classification, we’ll set up a Flask API endpoint where new call data can be sent for classification. This will enable you to classify calls as they come in.

After this in the bash code run the p1 app as : _python p1.py_
This will download the **pkl file** in the directory which will later be used to load the trained model in flask app **app.py.**

**Create a new Flask app named app.py :**

![image](https://github.com/user-attachments/assets/1320d33e-90ac-483a-b929-7b98071d2c88)

**Run the Flask app:**

in bash code type : _python app.py_
The app will deploy and start runnning

![image](https://github.com/user-attachments/assets/a3cfe1ca-3fc3-49cb-a841-2c404e3e1c46)

**Testing with curl**

Open a New Command Prompt Window

Do not close the command prompt where your Flask app is running.
**Run the Following Commands**

Here are some example commands you can use in the bash to test different categories. Replace the text in the JSON payload with the sentences you want to test.
_curl -X POST http://127.0.0.1:5000/classify -H "Content-Type: application/json" -d "{\"text\": \"I want to inquire about your latest product offerings.\"}"_

![image](https://github.com/user-attachments/assets/8f6a8b78-497a-4d60-abe1-de16d3f87dd3)
![image](https://github.com/user-attachments/assets/57a5123e-53dc-4f3f-b117-69d450c89c10)

**Happy Hacking :)**







