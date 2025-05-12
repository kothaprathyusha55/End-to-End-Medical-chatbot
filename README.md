# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mchatbot python=3.10 -y
```

```bash
conda activate mchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# Finally run the following command
python app.py
```




### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone


