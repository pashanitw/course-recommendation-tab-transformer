import pandas as pd
import torch
from transformer import TabTransformer
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F


data = pd.read_csv('./data.csv')
data.columns = map(str.lower, data.columns)
print(data.columns)
print(data.head())

categorical_columns = ['gender', 'stream', 'subject', 'course']
gender_values = data['gender'].unique().tolist()
stream_values = data['stream'].unique().tolist()
subject_values = data['subject'].unique().tolist()
target_values = data['course'].unique().tolist()


def show_course_id(id):
    print(f"Course Name:\n{data['course'][id]}",'\n')
    print(f"What is the course about?:\n{data['stream'][id]}",'\n')


data.reset_index(drop = True, inplace = True)
print(f'number of rows: {data.shape[0]}')
encoded_df = data.copy()
encoded_df[categorical_columns] = encoded_df[categorical_columns].apply(lambda x: x.factorize()[0])

mean = encoded_df['marks'].mean()
std = encoded_df['marks'].std()
encoded_df['marks'] = (encoded_df['marks'] - mean) / std

print(encoded_df.head())

# DATA LOADER
# Custom dataset class
class TabularDataset(Dataset):
    def __init__(self, data, target_col):
        self.data = data.drop(columns=[target_col, "marks"]).values
        self.target = data["course"].values  # Assign only the "marks" column to self.target
        self.continuous = data["marks"].values
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.continuous[idx], self.target[idx]

target_col = 'course'

encoded_df_tensor = encoded_df.applymap(lambda x: float(x))
dataset = TabularDataset(encoded_df_tensor, target_col)

# Split the dataset into training and validation sets
train_size = int(0.96 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(len(train_dataset), len(val_dataset))

# Create DataLoaders for training and validation sets
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = TabTransformer(
    categories = (len(gender_values), len(stream_values), len(subject_values)),
    num_continuous = 1,
    dim = 32,
    dim_out = len(target_values),
    depth = 2,
    heads = 4,
    attn_dropout = 0.1,
    ff_dropout = 0.1,
    mlp_hidden_mults = (4, 2),
    mlp_act = nn.ReLU(),
)
# Hyperparameters

lr = 0.001  # Learning rate
epochs = 50  # Number of epochs

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

def run():
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for categories, continuous, target in train_dataloader:
            continuous = continuous.unsqueeze(1)
            # Convert data and target to PyTorch tensors
            x_cat = torch.tensor(categories).to(torch.long)
            x_cont = continuous.to(torch.float32)
            target = target.to(torch.long)
            # Forward pass
            output = model(x_cat, x_cont)
            # Calculate loss
            loss = criterion(output, target)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for categories, continuous, target in val_dataloader:
                # Forward pass
                continuous = continuous.unsqueeze(1)
                # Convert data and target to PyTorch tensors
                x_cat = torch.tensor(categories).to(torch.long)
                x_cont = continuous.to(torch.float32)
                target = target.to(torch.long)
                output = model(x_cat, x_cont)
                print(output.shape, target.shape)

                # Calculate loss
                loss = criterion(output, target)
                val_loss += loss.item()

                # Calculate the number of correct predictions
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss /= len(val_dataloader)
        val_acc = correct / total
        train_loss /= len(train_dataloader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


run()
def tokenizer(record):
    gender, stream, subject, marks = record
    return torch.tensor([gender_values.index(gender), stream_values.index(stream), subject_values.index(subject)], dtype=torch.long), torch.tensor([(marks-mean)/std],dtype=torch.float32)


def infer():
    x_cat,x_cont = tokenizer(["female", "humanities", "history", 55])
    print(x_cat[None].shape, x_cont[None].shape)
    logits = model(x_cat[None], x_cont[None])
    preds = F.softmax(logits, dim=-1)
    print("=== predicted course ========", target_values[torch.argmax(preds, dim=-1).item()])


infer()
