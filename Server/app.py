from flask import Flask, request, jsonify, render_template
import util
import os

# Initialize the Flask application
app = Flask(__name__, template_folder='Client/templates', static_folder='Client')

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    try:
        total_sqft = float(request.form['total_sqft'])
        location = request.form['location']
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    estimated_price = util.get_estimated_price(location, total_sqft, bhk, bath)
    response = jsonify({
        'estimated_price': estimated_price
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    # Debugging print statements
    print("Starting Python Flask Server For Home Price Prediction...")
    print(f"Template folder: {os.path.abspath('Client/templates')}")
    print(f"Static folder: {os.path.abspath('Client')}")
    util.load_saved_artifacts()
    app.run(debug=True)
