from fastapi import FastAPI
from pathlib import Path
import torch
import uvicorn
from translation.config import get_config
from translation.model import get_model


config = get_config()
num_epochs = config['num_epochs']
BASE_PATH = Path(__file__).resolve().parent
MODEL_PATH = BASE_PATH / f'models/weights/tmodel_{num_epochs-1:02d}.pt'


# Create app
app = FastAPI(
    title='Translation Model API',
    description='API for machine translation',
)
# Load the tokenizers and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(str(MODEL_PATH), map_location=device)
tokenizer_src = checkpoint['tokenizer_src']
tokenizer_tgt = checkpoint['tokenizer_tgt']
model = get_model(
    config,
    tokenizer_src.get_vocab_size(),
    tokenizer_tgt.get_vocab_size()
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


@app.get("/")
def predict(text: str):
    result = model.translate(
        text,
        config,
        tokenizer_src,
        tokenizer_tgt,
        device
    )
    return {"translation": result}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0,', port='8000')
