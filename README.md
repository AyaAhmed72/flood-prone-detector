                                                 Flood Prone Area Detector
This project predicts whether a given location is flood-prone or not using Digital Elevation Model (DEM) .tif files in which It uses WhiteboxTools for terrain analysis, a Random Forest model, and the web 
interface is built with Streamlit.

Features

Upload a DEM .tif file and get a flood risk prediction.

Terrain preprocessing: depression filling, slope calculation, flow accumulation.

Extracts and processes statistical terrain features for prediction.

Displays results with confidence scores.

Project Workflow
Model Training & Preprocessing (Google Colab)

Load DEM datasets.

Extract features using WhiteboxTools.

Train a Random Forest Classifier.

Save the trained model as random_forest_model.pkl.

Run Web Interface (Visual Studio / Local Machine)

Install dependencies (shown below).

Run the Streamlit app:

 streamlit run c:\Users\AYA\Downloads\nu\NU.py → this will open the Flood Prone Area Detector in the browser.

Prediction

Upload a .tif DEM file.

The system will preprocess the DEM, extract features, and run the prediction.

Results will display:

Flood Prone or Not Flood Prone

Confidence Score in percentage.

Installation

1️. Clone Repository

git clone https://github.com/AyaAhmed72/flood-prone-detector.git

cd flood-prone-detector

2. Install Requirements
   
pip install -r requirements.txt

3️. Install WhiteboxTools

pip install whitebox

Usage

Place the trained random_forest_model.pkl in the project folder.

Run:
 streamlit run path_to_your_script/NU.py
 
Upload a .tif DEM file and view the prediction results.

📂 Project Structure

flood-prone-detector/

│── NU.py                  # Streamlit web interface

│── random_forest_model.pkl # Trained ML model

│── requirements.txt

│── README.md

Example Output

Flood Prone Area: 🌊 with high confidence.

Not Flood Prone Area: 🏞️ with confidence percentage.

You can have access to the data by sending an email to: aya72@aucegypt.edu




