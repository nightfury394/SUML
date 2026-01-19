
import subprocess
import sys

try:
    # Use python -m streamlit to be safe
    result = subprocess.run(
        [sys.executable, '-m', 'streamlit', 'run', 'test_app.py', '--server.headless', 'true'], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        timeout=5,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)
except subprocess.TimeoutExpired as e:
    # If it times out, it means it's running!
    print("Timeout Expired (App is running!)")
    # Decode bytes if needed (timeout expired stdout/stderr are bytes in some versions, but text=True should handle it)
    out = e.stdout if e.stdout else ""
    err = e.stderr if e.stderr else ""
    if isinstance(out, bytes): out = out.decode()
    if isinstance(err, bytes): err = err.decode()
    
    print("STDOUT:", out)
    print("STDERR:", err)
except Exception as e:
    print(f"An error occurred: {e}")
