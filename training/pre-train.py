import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 1 # regression
model1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer1 = AutoTokenizer.from_pretrained(model_name)

print(f'model 1: {model_name}')
print(f'model1 | tokenizer1')

model_name = "pythainlp/thainer-corpus-v2-base-model"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 1 # regression
model2 = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer2 = AutoTokenizer.from_pretrained(model_name)
      
print(f'model 2: {model_name}')
print(f'model2 | tokenizer2')


model_name = "KoichiYasuoka/roberta-base-thai-spm-upos"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 1 # regression
model3 = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer3 = AutoTokenizer.from_pretrained(model_name)
      
print(f'model 3: {model_name}')
print(f'model3 | tokenizer3')

loss_fn = torch.nn.L1Loss() # MAE loss

device = torch.device("cpu") # Move your model to the device (CPU)

gradient_accumulation_steps = 4 # Set up gradient accumulation

batch_size = 16  # Reduced batch size
num_epochs = 1