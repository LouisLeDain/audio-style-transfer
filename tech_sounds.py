## import libraries

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf

## define gpu 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## sound processing function and model definition

# Load audio files
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Compute the spectrogram
def compute_spectrogram(y):
    return librosa.stft(y)

# Reconstruct audio from spectrogram
def reconstruct_audio(spectrogram, sr):
    y = librosa.istft(spectrogram)
    return y

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(28306400, 16)
        self.fc2 = nn.Linear(16, 1025*863)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 28306400)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 1, 1025, 863)
        return x

## Load content and style audio

content_audio, sr = load_audio('input_content.mp3')
style_audio, _ = load_audio('input_style.mp3')


# Compute spectrograms
content_spectrogram = compute_spectrogram(content_audio)
style_spectrogram = compute_spectrogram(style_audio)

# Convert spectrograms to tensors
content_tensor = torch.tensor(np.abs(content_spectrogram), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
style_tensor = torch.tensor(np.abs(style_spectrogram), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Initialize the model and optimizer
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Device selection

# Move tensors and model to the specified device
content_tensor = content_tensor.to(device)
style_tensor = style_tensor.to(device)
model = model.to(device)

## define loss function

# content loss

def content_loss(content_tensor, generated_tensor):
    return torch.mean((content_tensor - generated_tensor)**2)

# style loss

def gram_matrix(input):
    '''
    Entrée : un tenseur (a, b, c, d)
    Sortie : un tenseur (a*b, a*b)
    '''
    input = input.squeeze(0)
    input = input.squeeze(0)
    gram = torch.matmul(input, input.t())
    return gram/(input.shape[0]*input.shape[1])

def style_loss(style, gen):
    gram_gen = gram_matrix(gen)
    gram_style = gram_matrix(style)
    return content_loss(gram_gen,gram_style)

def total_loss(content_tensor, style_tensor, generated_tensor, alpha=1, beta=1000):
    return alpha*content_loss(content_tensor, generated_tensor) + beta*style_loss(style_tensor, generated_tensor)

## Training loop
num_epochs = 1
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    output = model(content_tensor)
    print(output.shape)
    loss = total_loss(content_tensor, style_tensor, output)
    loss.backward()
    
    optimizer.step()
    #print(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.4f}')

# Reconstruct audio from the output spectrogram
output_spectrogram = output.detach().cpu().numpy().squeeze()
#print(output_spectrogram.shape)
reconstructed_audio = reconstruct_audio(output_spectrogram, sr)

# Save the reconstructed audio
sf.write('output.mp3', reconstructed_audio, sr)