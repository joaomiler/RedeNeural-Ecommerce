# E-commerce Recommendation System

A web application that displays user profiles and product listings, featuring an integrated machine learning recommendation system powered by TensorFlow.js. The system analyzes user behavior and purchase history to generate intelligent product recommendations in real time.

## Project Structure

- index.html - Main HTML file
- index.js - Application entry point
- view/ - DOM management and templates
- controller/ - Controllers (views and services)
- service/ - Business logic and machine learning integration
- data/ - JSON data (users and products)
- ml/ - TensorFlow models and recommendation logic

## Setup and Run

1. Install dependencies
   npm install

2. Start the application
   npm start

3. Open in browser
   http://localhost:3000

## Features

- User profile selection with detailed information
- Purchase history tracking
- Product listing with "Buy Now" functionality
- Real-time recommendation engine using TensorFlow.js
- Intelligent suggestions based on user behavior
- Session-based data tracking

## Machine Learning

The application includes a machine learning layer built with TensorFlow.js, responsible for:

- Learning user preferences from purchase history
- Generating personalized product recommendations
- Identifying patterns and similarities between users
- Continuously improving recommendations as new data is collected

## Tech Stack

- JavaScript
- TensorFlow.js
- HTML5 / CSS3

## Notes

This project demonstrates how machine learning can be applied directly in the frontend using TensorFlow.js, enabling real-time recommendations without requiring a backend ML service.
