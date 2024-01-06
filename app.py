from flask import Flask, render_template, request, redirect, url_for, flash
from model.GPT_Model import BigramLanguageModel, chars
import torch
model = BigramLanguageModel()

stoi = {ch:i for i,ch in enumerate(chars)}
itoi = {i:ch for i,ch in enumerate(chars)}
encoder = lambda x: [stoi[ch] for ch in x]
decoder = lambda x: ''.join([itoi[i] for i in x])


device = torch.device('cpu')
# load model 
model.load_state_dict(torch.load('model/model.pt', map_location=device))

# device will be mps or cuda or cpu
# if torch.has_mps:
#     device = torch.device('mps')
# elif torch.cuda.is_available():
#     device = torch.device('cuda')
# else:

# context = str(input('Enter Context: '))
# context = encoder(context)
# context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension
m = model.to(device)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        context_str = request.form['context']
        context = encoder(context_str)
        context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        generated_tokens = m.generate(idx=context, max_new_tokens=1000)[0].tolist()
        decoded_tokens = ""
        
        for token in generated_tokens:
            decoded_tokens += decoder([token])
            
        
        return decoded_tokens
    
    
if __name__ == '__main__':
    app.run(debug=True, port=8000)
        
