import schedule
import time
import subprocess
from datetime import datetime, timedelta
import socket
from tqdm import tqdm

def run_command():
    node = str(socket.gethostname())
    command = f"salloc --partition=GPU-8A100 --gres=gpu:7 --nodes=1 --ntasks=1 --time=5-00:00:00 --nodelist={node} --qos=gpu_8a100"
    subprocess.run(command, shell=True)

# Calculate the time difference to 8:00 AM today
now = datetime.now()
target_time = now.replace(hour=8, minute=0, second=0, microsecond=0)

# If it's already past 8:00 AM today, schedule it for tomorrow
if now > target_time:
    target_time += timedelta(days=1)

# Calculate the total seconds until the task should run
total_seconds = int((target_time - now).total_seconds())

# Schedule the task
schedule.every().day.at(target_time.strftime("%H:%M")).do(run_command)

print(f"Task scheduled for {target_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Progress bar loop
for remaining in tqdm(range(total_seconds), desc="Time remaining", unit="s"):
    time.sleep(1)
    now = datetime.now()
    schedule.run_pending()

# Task will be executed when the progress bar completes
run_command()
