import torch
import time

from GPT_Model import BigramLanguageModel, chars





model = BigramLanguageModel()

stoi = {ch:i for i,ch in enumerate(chars)}
itoi = {i:ch for i,ch in enumerate(chars)}
encoder = lambda x: [stoi[ch] for ch in x]
decoder = lambda x: ''.join([itoi[i] for i in x])

# load model 
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))

# device will be mps or cuda or cpu
# if torch.has_mps:
#     device = torch.device('mps')
# elif torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
device = torch.device('cpu')
context = str(input('Enter Context: '))
context = encoder(context)
context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension
m = model.to(device)
# # print generated tokens letter by letter
# context = torch.zeros((1,1), dtype=torch.long, device=device)
print("Thinking..... \n")
generated_tokens = m.generate(idx=context, max_new_tokens=1000)[0].tolist()
for token in generated_tokens:
    print(decoder([token]), end='', flush=True)
    time.sleep(0.01)
# print 'generated tokens letter by letter