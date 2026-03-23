# E-commerce Recommendation System

A web application that displays user profiles and product listings, featuring an integrated machine learning recommendation system powered by TensorFlow.js. The system analyzes user behavior and purchase history to generate intelligent product recommendations in real time.

# Project Structure #
index.html - Main HTML file for the application
index.js - Entry point for the application
view/ - Contains classes for managing the DOM and templates
controller/ - Contains controllers to connect views and services
service/ - Contains business logic and machine learning integration
data/ - Contains JSON files with user and product data
ml/ (optional, se existir) - TensorFlow models and recommendation logic
Setup and Run

# Install dependencies:
npm install
Start the application:
npm start
Open your browser and navigate to http://localhost:8080

# Features
User profile selection with detailed information
Purchase history tracking and visualization
Product listing with "Buy Now" functionality
Real-time recommendation engine using TensorFlow.js
Intelligent suggestions based on user behavior and past purchases
Session-based and model-based data tracking

# Machine Learning
The application includes a machine learning layer built with TensorFlow.js, responsible for:
Learning user preferences from purchase history
Generating personalized product recommendations
Identifying patterns and similarities between users
Continuously improving recommendations as new data is collected

