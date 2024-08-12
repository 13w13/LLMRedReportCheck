# LLM-RedReportCheck

LLM-RedReportCheck is an AI-powered tool designed to validate humanitarian data reports. It uses Large Language Models to cross-check narrative reports against numerical data, identifying inconsistencies and generating validation questions.

## Features

- Upload narrative reports and data files
- Analyze reports using state-of-the-art language models
- Generate validation questions to highlight potential inconsistencies
- Store reports and data securely in Azure Blob Storage

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your .env file with Azure and Hugging Face credentials
4. Run the application: `uvicorn src.main:app --reload`

## Usage

Send a POST request to `/validate` with two files:
- `narrative`: A Word document containing the narrative report
- `data`: An Excel file containing the numerical data

The API will return a list of validation questions based on the analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.