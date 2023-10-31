import torch
import torchvision
import torchvision.transforms as transforms
from vit_pytorch.vit import ViT

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transforms.ToTensor()
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # ViT-Tiny based settings
    image_size, in_channels = 32, 3
    patch_size = 4
    embed_dim = 192
    num_layers = 12
    mlp_dim = 768
    num_heads = 3
    drop_p = 0.5
    num_classes = 10

    net = ViT(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=patch_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        drop_p=drop_p,
        num_classes=num_classes,
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):
        for i, (images, labels) in enumerate(trainloader):
            # Forward propagation
            outputs = net(images)
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the progress of the training
        if epoch % 1 == 0:
            print("Epoch: {} Loss: {:.4f}".format(epoch, loss.item()))

    # Evaluate the model with the test dataset
    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("Accuracy: {:.2f}%".format(100 * correct / total))
