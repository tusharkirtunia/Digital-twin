from celery.schedules import crontab
from app.workers.tasks import celery_app

celery_app.conf.beat_schedule = {
    "weekly-report-every-monday": {
        "task": "app.workers.tasks.generate_weekly_report",
        "schedule": crontab(hour=8, minute=0, day_of_week=1),
        "args": [],   # In production: loop over all user IDs
    },
}
celery_app.conf.timezone = "UTC"