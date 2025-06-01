from app import create_app
from app.services.daily_tasks import start_daily_tasks_thread

app = create_app()

# Start daily tasks in a separate thread
start_daily_tasks_thread()

if __name__ == '__main__':
    app.run(debug=True)
