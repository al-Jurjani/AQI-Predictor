import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()
api = os.getenv("hopsworks_api_key")

project = hopsworks.login(api_key_value=api)  # connects to my hopsworks project
fs = project.get_feature_store()  # gets the feature store object of my project

print("Connected to hopsworks project: ", project.name)
print("Feature store name: ", fs.name)
