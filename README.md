The project in this repository has been developed for the purposes of ML DevOps course from udacity. The project is tiled as "Deploying a Machine Learning Model on Heroku with FastAPI", and the starter code that has been used can be found at: https://github.com/udacity/nd0821-c3-starter-code. 

The starter code has been used at the commit with SHA `e1662e831eca8318a12b261ea5808245a75a6872`. And instructions on how to start can be found at the associated README.

# Project

There are several components that go into this project, the 3 most basic being the development of the model, the REST API, and the deployment process. Tests have been developed for the first two, and they are combined in a deployment process that leverages github actions and heroku's github integration.

## Model

A Random Forest model was trained with the provided data. Before the training, the data were cleaned, and the clean version was pushed to dvc. Afterwards, two transformers were used to Binarize labels and to OneHotEncode categorical features. Simple tests for the methods that make these transformations have also been developed.

## API

A FastAPI application is developed to serve the model via an API. The API contains two simple endpoints: a GET one in the root to mainly check the status of the deployment, and a POST one at `/predict/` that can be used for accessing the inference functionality. The body of the POST request needs to specify a data point with the required arguments that can be found at the `/docs` or the accompanying model card. 

Finally, simple tests that verify the expected responses for both endpoints are created. In the case of POST there are 2 tests, one for each possible label.

docs: `https://ml-model-ci-cd.herokuapp.com/docs`

## Deployment

The deployment is on heruku which is connected to github and deploys the app in every push after the CI workflow ran successfully. For it to succeed, several flake8 checks and unit tests need to pass. The API is hosted at the following base url: `https://ml-model-ci-cd.herokuapp.com/`