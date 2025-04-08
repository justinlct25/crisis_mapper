# HumanAI Social Media Crisis Detection

This project is designed to detect and analyze crises on social media platforms using AI-powered tools.

## Folder Structure

Ensure the following folder structure exists in your project directory before running the pipeline:

```
humanai_social_media_crisis_detection/
├── output/               # Directory where all output files will be stored
├── .env                  # Environment variables file
├── README.md             # Project documentation
├── main.py               # Main script to run the pipeline
└── other_project_files/  # Other necessary files and scripts
```

## Environment Variables

Create a `.env` file in the root directory of the project with the following variables:

```env
REDDIT_USERNAME=<your_reddit_username>
REDDIT_PASSWORD=<your_reddit_password>
REDDIT_CLIENT_ID=<your_reddit_client_id>
REDDIT_CLIENT_SECRET=<your_reddit_client_secret>
OPENAI_API_KEY=<your_openai_api_key>
```

Replace `<your_reddit_username>`, `<your_reddit_password>`, `<your_reddit_client_id>`, `<your_reddit_client_secret>`, and `<your_openai_api_key>` with your actual credentials.

## Running the Pipeline

To run the pipeline, execute the following command in your terminal:

```bash
python pipeline_function_runner.py
```

This will start the process of collecting data from Reddit and analyzing it using OpenAI's API.

## Output

All generated outputs, such as processed data and analysis results, will be saved in the `output/` directory.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install via `requirements.txt` if available)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/humanai_social_media_crisis_detection.git
    cd humanai_social_media_crisis_detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the `.env` file as described above.

You're now ready to run the pipeline!