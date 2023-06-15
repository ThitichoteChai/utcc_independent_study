import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 1 # regression
model1 = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer1 = AutoTokenizer.from_pretrained(model_name)

loss_fn = torch.nn.L1Loss() # MAE loss
device = torch.device("cpu") # Move your model to the device (CPU)
gradient_accumulation_steps = 4 # Set up gradient accumulation
batch_size = 16  # Reduced batch size
num_epochs = 1

model = model1
tokenizer = tokenizer1

model.to(device)

# Define your optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Tokenize data and create dataloader
encoded_data = tokenizer.batch_encode_plus(
    train_data['comment'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=100,
    return_tensors='pt'
)

labels = torch.tensor(train_data['score'].values, dtype=torch.float32)  # Convert labels to float32
dataset = TensorDataset(encoded_data['input_ids'], encoded_data['attention_mask'], labels)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Preprocess the test data
encoded_test_data = tokenizer.batch_encode_plus(
    test_data['comment'].tolist(),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

test_labels = torch.tensor(test_data['score'].values, dtype=torch.float32)
test_dataset = TensorDataset(encoded_test_data['input_ids'], encoded_test_data['attention_mask'], test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Fine-tune the model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    step = 0

    for batch in train_dataloader:
        step += 1
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps

        loss.backward()

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            epoch_loss += loss.item()

    print(f'Epoch: {epoch + 1}, Loss: {epoch_loss / step}')

# Evaluate the model's predictions against the actual values
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = logits.squeeze(1).cpu().numpy()

        predictions.extend(preds)
        actuals.extend(labels.cpu().numpy())

mae = mean_absolute_error(actuals, predictions)
print(f'Mean Absolute Error: {mae:.6f}')

result_df = pd.DataFrame({'actual': actuals, 'predicted': predictions})
result_df['comment'] = test_data['comment']
result_df = result_df[['comment', 'actual', 'predicted']].sort_values(by='predicted', ascending=False)

my_result = result(result_df)