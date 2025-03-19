# MedGPT - Medical Symptom Analysis & Doctor Recommendation

![MedGPT Logo](https://img.shields.io/badge/MedGPT-🧠-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange)
![LangChain](https://img.shields.io/badge/LangChain-0.0.311-green)

## Overview

MedGPT is a conversational AI assistant built on AWS Bedrock and LangChain that helps users analyze their medical symptoms and recommends nearby doctors in Chennai. The application uses a step-by-step conversational approach to understand symptoms, provides information about potential causes, and suggests appropriate medical professionals.

![MedGPT Screenshot](https://user-images.githubusercontent.com/yourUsername/your-repo/assets/screenshot.png)

## Features

- 🩺 **Symptom Analysis**: Conversational AI that asks targeted questions to understand user symptoms
- 📊 **Medical Knowledge Base**: Built-in medical information to provide accurate responses
- 🏥 **Doctor Recommendations**: Suggests nearby doctors based on user location in Chennai
- ⭐ **Rating System**: Recommends doctors based on ratings and location
- 🔄 **Model Selection**: Switch between different LLM models (Llama 3, Mistral, DeepSeek)
- 🧩 **Customizable Parameters**: Adjust chunk size and overlap for knowledge base processing

## Tech Stack

- **Streamlit**: Frontend interface
- **AWS Bedrock**: Provides LLM capabilities (Llama 3, Mistral, etc.)
- **LangChain**: Framework for building LLM applications
- **Boto3**: AWS SDK for Python
- **Semantic Search**: Uses embeddings to find relevant medical information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medgpt.git
cd medgpt
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your AWS credentials (options):
   - Create a `.env` file with your AWS credentials
   - Configure them in `app.py`
   - Use AWS CLI to configure credentials

## Configuration

1. Update the `knowledge_base.txt` file with your medical information
2. Update the `doctor_list.txt` file with doctor information in the format:
   ```
   Doctor Name | Specialty | Area | Rating
   ```

3. AWS credentials can be set in the `.env` file:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Interact with the chatbot by describing your symptoms

4. When prompted, provide your location in Chennai to get doctor recommendations

## Project Structure

```
medgpt/
├── app.py                  # Main Streamlit application
├── knowledge_base.txt      # Medical knowledge database
├── doctor_list.txt         # List of doctors with locations and ratings
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not tracked by git)
├── .gitignore              # Git ignore file
└── README.md               # Project documentation
```

## AWS Bedrock Models

The application supports the following models:
- meta.llama3-70b-instruct-v1:0
- meta.llama3-1-70b-instruct-v1:0
- mistral.mixtral-8x7b-instruct-v0:1
- mistral.mistral-7b-instruct-v0:2
- deepseek.r1-v1:0

## Customization

- **Model Parameters**: Adjust temperature, max tokens, and top_p in the code
- **Chunk Size**: Control how the knowledge base is split for retrieval
- **Prompt Template**: Modify the conversation flow in the PROMPT_TEMPLATE variable

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AWS Bedrock for providing the LLM capabilities
- LangChain for the framework
- All contributors and maintainers
