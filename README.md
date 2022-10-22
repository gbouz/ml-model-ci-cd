The project in this repository has been developed for the purposes of ML DevOps course from udacity. The project specification is "Deploying a Machine Learning Model on Heroku with FastAPI", and starter code has been used that can be found at: https://github.com/udacity/nd0821-c3-starter-code. 

The starter code has been used at SHA `e1662e831eca8318a12b261ea5808245a75a6872`. And in the corresponding README there are instructions on how to start.

# Project

There are several components into the project that are explained here. The basis is that a model is being trained, and unit tests were developed to make sure that it remains relevant in how the underlying data distribution shapes in the future. The model is deployed as a web application and can be accessed via its REST API. Tests for each endpoint have been also been developed, and invoked automatically by a CI/CD process that conditions the deployment process.

## Model

A Random Forest model was trained with the provided data. Tests for the shape of the data have also been created.

## API

A FastAPI application is developed to serve the model via an API. Tests for the endpoint that delivers the prediction are created.

## Deployment

The deployment is on heruku which is connected to github and runs the CI/CD workflows specified in the repo.