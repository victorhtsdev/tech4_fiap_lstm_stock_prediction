from app import create_app
from app.services.daily_tasks import start_daily_tasks_thread
from app.routes.api_routes import swaggerui_blueprint, bp

app = create_app()

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')

# Register API routes blueprint with a unique name
app.register_blueprint(bp, name='api_routes')

# Start daily tasks in a separate thread
start_daily_tasks_thread()

if __name__ == '__main__':
    app.run(debug=True)
