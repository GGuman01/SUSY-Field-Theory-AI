# training.py

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from susy_model import SUSYModel, susy_loss

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Initialize the model
input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels, 3 channels (RGB)
num_classes = 10  # CIFAR-10 has 10 classes
model = SUSYModel(input_size, num_classes)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define function to generate adversarial examples using FGSM
def fgsm_attack(data, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_data = data + epsilon * sign_data_grad
    return torch.clamp(perturbed_data, 0, 1)

# Define function for PGD attack (multiple iterations of FGSM)
def pgd_attack(data, epsilon, alpha, num_iter, model, label):
    perturbed_data = data.clone().detach()
    for i in range(num_iter):
        perturbed_data.requires_grad = True
        outputs = model(perturbed_data)
        loss = F.cross_entropy(outputs, label)
        model.zero_grad()
        loss.backward()
        grad = perturbed_data.grad
        perturbed_data = perturbed_data + alpha * grad.sign()
        perturbed_data = torch.clamp(perturbed_data, data - epsilon, data + epsilon)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

# Training loop with adversarial training (FGSM)
epsilon = 0.03  # FGSM perturbation size
alpha = 0.01    # PGD step size
pgd_iter = 10   # Number of iterations for PGD

for epoch in range(10):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs.requires_grad = True
        inputs = inputs.view(inputs.size(0), -1)  # Flatten the image data

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Create adversarial examples with FGSM and PGD
        loss = susy_loss(outputs, labels, outputs)
        loss.backward()
        optimizer.step()

        # For robustness, create adversarial examples using FGSM
        gradients = inputs.grad
        perturbed_inputs_fgsm = fgsm_attack(inputs, epsilon, gradients)

        # PGD attack for stronger adversarial examples
        perturbed_inputs_pgd = pgd_attack(inputs, epsilon, alpha, pgd_iter, model, labels)

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Test the robustness on the test set using FGSM and PGD attacks
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(images.size(0), -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on clean test images: {100 * correct / total:.2f}%')

# Adversarial accuracy using FGSM
correct_fgsm = 0
total_fgsm = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images.requires_grad = True
        images = images.view(images.size(0), -1)
        
        # Get the adversarial example using FGSM
        gradients = images.grad
        perturbed_images_fgsm = fgsm_attack(images, epsilon, gradients)
        
        outputs = model(perturbed_images_fgsm)
        _, predicted = torch.max(outputs.data, 1)
        total_fgsm += labels.size(0)
        correct_fgsm += (predicted == labels).sum().item()

print(f'Accuracy on FGSM adversarial test images: {100 * correct_fgsm / total_fgsm:.2f}%')

# Adversarial accuracy using PGD
correct_pgd = 0
total_pgd = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(images.size(0), -1)
        
        # Get the adversarial example using PGD
        perturbed_images_pgd = pgd_attack(images, epsilon, alpha, pgd_iter, model, labels)
        
        outputs = model(perturbed_images_pgd)
        _, predicted = torch.max(outputs.data, 1)
        total_pgd += labels.size(0)
        correct_pgd += (predicted == labels).sum().item()

print(f'Accuracy on PGD adversarial test images: {100 * correct_pgd / total_pgd:.2f}%')
