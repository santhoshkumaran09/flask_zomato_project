from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
import traceback

app = Flask(__name__)

def prepare_data(data):
    try:
        locations = []
        for location in data.get('locations', []):
            loc = location['location']
            locations.append({
                'area': loc['area'],
                'population_density': loc['population_density'],
                'avg_income': loc['avg_income'],
                'existing_restaurants': loc['existing_restaurants'],
                'competition_score': loc['competition_score'],
                'accessibility_score': loc['accessibility_score'],
                'delivery_speed_score': loc.get('delivery_speed_score', 0),
                'existing_delivery_apps': loc.get('existing_delivery_apps', []),
                'current_profit': loc.get('current_profit', 0)
            })
        return pd.DataFrame(locations)
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        print(traceback.format_exc())
        raise

def train_model(df):
    try:
        # Features for prediction
        features = ['population_density', 'avg_income', 'existing_restaurants', 
                   'competition_score', 'accessibility_score', 'delivery_speed_score']
        X = df[features]
        y = df['current_profit']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, features
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        print(traceback.format_exc())
        raise

def analyze_location_potential(df, model, features):
    try:
        # Get predictions
        predictions = model.predict(df[features])
        
        # Calculate potential score (0-100)
        max_profit = float(df['current_profit'].max())
        potential_scores = (predictions / max_profit) * 100
        
        # Calculate market saturation score
        saturation_scores = 100 - (df['existing_restaurants'] / float(df['existing_restaurants'].max()) * 100)
        
        # Calculate delivery speed potential
        speed_scores = df['delivery_speed_score']
        
        # Calculate competition impact
        competition_impact = 100 - (df['competition_score'] / float(df['competition_score'].max()) * 100)
        
        # Calculate income potential
        income_scores = (df['avg_income'] / float(df['avg_income'].max())) * 100
        
        # Calculate population potential
        population_scores = (df['population_density'] / float(df['population_density'].max())) * 100
        
        # Combine scores with weights
        # 35% profit potential, 15% market saturation, 15% delivery speed, 
        # 15% competition impact, 10% income potential, 10% population potential
        final_scores = (
            potential_scores * 0.35 + 
            saturation_scores * 0.15 + 
            speed_scores * 0.15 + 
            competition_impact * 0.15 + 
            income_scores * 0.10 + 
            population_scores * 0.10
        )
        
        # Convert NumPy arrays to Python lists
        return predictions.tolist(), final_scores.tolist(), {
            'potential_scores': potential_scores.tolist(),
            'saturation_scores': saturation_scores.tolist(),
            'speed_scores': speed_scores.tolist(),
            'competition_impact': competition_impact.tolist(),
            'income_scores': income_scores.tolist(),
            'population_scores': population_scores.tolist()
        }
    except Exception as e:
        print(f"Error in analyze_location_potential: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print('Received request to /analyze')
        if 'file' not in request.files:
            print('No file part in request.files')
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = request.files['file']
        print('File received:', file.filename)
        if file.filename == '':
            print('No file selected')
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        if not file.filename.endswith('.json'):
            print('File is not a JSON file')
            return jsonify({
                'success': False,
                'error': 'Please upload a JSON file'
            })
        
        # Read and parse JSON data
        try:
            data = json.load(file)
            print('JSON loaded successfully')
        except json.JSONDecodeError as e:
            print('JSON decode error:', str(e))
            return jsonify({
                'success': False,
                'error': f'Invalid JSON format: {str(e)}'
            })
        
        # Prepare data
        df = prepare_data(data)
        print('DataFrame prepared:', df.shape)
        
        if df.empty:
            print('DataFrame is empty')
            return jsonify({
                'success': False,
                'error': 'No valid location data found in the file'
            })
        
        # Train model
        model, features = train_model(df)
        print('Model trained')
        
        # Analyze location potential
        predictions, scores, detailed_scores = analyze_location_potential(df, model, features)
        print('Analysis complete')
        
        # Calculate average values for comparison
        avg_income = float(df['avg_income'].mean())
        avg_population = float(df['population_density'].mean())
        avg_competition = float(df['competition_score'].mean())
        
        # Combine results
        results = []
        for i, location in enumerate(df.to_dict('records')):
            # Calculate recommendation based on multiple factors
            recommendation = 'High Potential' if scores[i] >= 70 else 'Medium Potential' if scores[i] >= 40 else 'Low Potential'
            
            # Generate detailed insights
            insights = []
            
            # Profit potential insights
            if predictions[i] > float(df['current_profit'].mean()):
                insights.append(f"High profit potential: ₹{predictions[i]:,.0f} monthly")
            
            # Competition insights
            if len(location.get('existing_delivery_apps', [])) < 3:
                insights.append("Low competition from delivery apps")
            if float(location['competition_score']) < avg_competition:
                insights.append("Below average competition in the area")
            
            # Market insights
            if float(location['population_density']) > avg_population:
                insights.append("High population density for customer base")
            if float(location['avg_income']) > avg_income:
                insights.append("High average income for premium pricing")
            
            # Delivery insights
            if float(location.get('delivery_speed_score', 0)) >= 80:
                insights.append("Excellent delivery speed potential")
            
            # Market saturation insights
            if float(location['existing_restaurants']) < float(df['existing_restaurants'].mean()):
                insights.append("Less saturated market with growth potential")
            
            # Add profit-focused recommendations
            profit_recommendations = []
            if float(location['avg_income']) > avg_income:
                profit_recommendations.append("Consider premium pricing strategy")
            if float(location['population_density']) > avg_population:
                profit_recommendations.append("High volume potential for delivery orders")
            if float(location.get('delivery_speed_score', 0)) >= 80:
                profit_recommendations.append("Fast delivery capability can command premium pricing")
            
            results.append({
                'area': location['area'],
                'population_density': float(location['population_density']),
                'avg_income': float(location['avg_income']),
                'existing_restaurants': int(location['existing_restaurants']),
                'competition_score': float(location['competition_score']),
                'accessibility_score': float(location['accessibility_score']),
                'delivery_speed_score': float(location.get('delivery_speed_score', 0)),
                'existing_delivery_apps': location.get('existing_delivery_apps', []),
                'predicted_profit': float(round(predictions[i], 2)),
                'potential_score': float(round(scores[i], 2)),
                'recommendation': recommendation,
                'insights': insights,
                'profit_recommendations': profit_recommendations,
                'detailed_scores': {
                    'profit_potential': float(round(detailed_scores['potential_scores'][i], 2)),
                    'market_saturation': float(round(detailed_scores['saturation_scores'][i], 2)),
                    'delivery_speed': float(round(detailed_scores['speed_scores'][i], 2)),
                    'competition_impact': float(round(detailed_scores['competition_impact'][i], 2)),
                    'income_potential': float(round(detailed_scores['income_scores'][i], 2)),
                    'population_potential': float(round(detailed_scores['population_scores'][i], 2))
                }
            })
        
        # Sort results by potential score
        results.sort(key=lambda x: x['potential_score'], reverse=True)
        print('Results ready, sending response')
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 