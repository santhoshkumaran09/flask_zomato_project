# Restaurant Rating Predictor

This Flask application analyzes restaurant data from JSON files and predicts ratings based on various features. It provides a user-friendly interface to view and compare actual vs. predicted ratings for restaurants.

## Features

- Upload restaurant data in JSON format
- Analyze restaurant ratings and features
- Predict restaurant ratings using machine learning
- Modern and responsive UI with drag-and-drop file upload
- Real-time data analysis

## Setup Instructions

1. Clone this repository
2. Install the required dependencies:
   pip install -r requirements.txt
   
3. Run the application:
   python app.py
  
4. Open your browser and go to `http://localhost:5000`

## Usage

1. Prepare your restaurant data in JSON format with the following structure:
   ```json
   {
     "restaurants": [
       {
         "restaurant": {
           "name": "Restaurant Name",
           "cuisines": "Cuisine Type",
           "average_cost_for_two": 1000,
           "user_rating": {
             "aggregate_rating": "4.5",
             "votes": "1000"
           }
         }
       }
     ]
   }
   ```

2. Upload your JSON file using the web interface:
   - Drag and drop your JSON file onto the upload area, or
   - Click "Choose File" to select your JSON file

3. Click "Analyze Data" to process the file

4. View the results showing:
   - Restaurant name
   - Cuisines
   - Cost for two
   - Actual rating
   - Predicted rating

## Technologies Used

- Flask
- scikit-learn
- pandas
- Bootstrap
- JavaScript 
