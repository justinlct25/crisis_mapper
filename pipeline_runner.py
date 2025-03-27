import subprocess
import sys

def run_script(script_name, args=[]):
    """Run a Python script with optional arguments."""
    try:
        print(f"\nRunning {script_name} with arguments: {' '.join(args)}...")
        subprocess.run([sys.executable, script_name] + args, check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        sys.exit(1)

def main():
    # Check command-line arguments
    if len(sys.argv) != 2 or sys.argv[1] not in ['r', 'x']:
        print("Usage: python pipeline_runner.py [r|x]")
        print("r: Run pipeline for Reddit data")
        print("x: Run pipeline for X.com data")
        sys.exit(1)

    source = sys.argv[1]

    # Step 1: Run extractor script
    run_script("extractor_reddit_keywords.py" if source == 'r' else "extractor_x_keywords.py")

    # Step 2: Run classifier script
    run_script("classifier_bert.py", [source])

    # Step 3: Run geolocator script
    run_script("geolocator_gpt.py", [source])

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()