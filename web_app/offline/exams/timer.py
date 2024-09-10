from datetime import datetime, timedelta

from bson import ObjectId

from ..repository import examRespository
from ..config import AIConstant

class ExamTimer:
    def __init__(self):
        self.start_time = None
        self.duration = None

    def time_to_datetime(self, t, base_date):
        """Combine a time object with a base date to create a datetime object."""
        return datetime.combine(base_date, t)

    def evaluate_time(self):
        """Check if the exam duration has elapsed."""
        if self.start_time is None or self.duration is None:
            raise ValueError("Timer has not been started or duration is not set.")
        
        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        val = elapsed_time > self.duration
        return val

    def start_exam_timer(self, exam_id):
        """Initialize the timer based on exam start and end times."""
        exam = examRespository.find_one({"_id": ObjectId(exam_id)})
        
        # Extract start and end time from the database
        start_time_str = exam["start_time"]
        end_time_str = exam["end_time"]

        # Define the base date (current date)
        base_date = datetime.now().date()

        # Parse the start and end times into datetime.time objects
        start_time = datetime.strptime(start_time_str, AIConstant.time_format).time()
        end_time = datetime.strptime(end_time_str, AIConstant.time_format).time()

        # Combine base date with time objects to create datetime objects
        start_datetime = self.time_to_datetime(start_time, base_date)
        end_datetime = self.time_to_datetime(end_time, base_date)

        # Set the duration and start time
        self.duration = end_datetime - start_datetime
        self.start_time = datetime.now()
