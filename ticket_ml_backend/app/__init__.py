from flask import Flask
from flask_cors import CORS
from flask_smorest import Api

from .startup import register_blueprints

# Initialize Flask app with CORS and OpenAPI configuration
app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app, resources={r"/*": {"origins": "*"}})

# API/OpenAPI metadata
app.config["API_TITLE"] = "Ticket ML Backend API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_JSON_PATH"] = "openapi.json"
app.config['OPENAPI_URL_PREFIX'] = '/docs'
app.config["OPENAPI_SWAGGER_UI_PATH"] = ""
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

# Create Api and register blueprints
api = Api(app)
register_blueprints(api)
