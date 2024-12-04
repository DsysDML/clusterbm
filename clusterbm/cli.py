import os
import subprocess
import sys


def main():

    # Get the directory of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT = "maketree.py"

    # Run the corresponding Python script with the remaining optional arguments
    script_path = os.path.join(SCRIPT_DIR, SCRIPT)
    proc = subprocess.call(
        [sys.executable, script_path] + sys.argv[1:],
    )
    
if __name__ == '__main__':
    main()