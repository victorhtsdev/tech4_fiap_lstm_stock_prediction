from flask import Flask
from app.routes.api_routes import bp as api_bp
from app.utils.status_tracker import training_status

# Clear training_status on server start
training_status.clear()

def create_app():
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix='/api')
    return app
