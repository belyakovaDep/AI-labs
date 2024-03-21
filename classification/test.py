import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tf
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    batch = 32
    model = tv.models.resnet34()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = tf.Compose([tf.Resize(256), tf.ToTensor(),
                            tf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.225, 0.225, 0.225]), 
                            tf.CenterCrop(224)])

    orgset = tv.datasets.ImageFolder('./simpsons_dataset', transform = transform)

    classes = orgset.classes

    split_indices = torch.load('split_indices.pth')

    train_set = torch.utils.data.Subset(orgset, split_indices['train_indices'])
    test_set = torch.utils.data.Subset(orgset, split_indices['test_indices'])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch, shuffle=True, num_workers=4)

    model.fc = nn.Linear(model.fc.in_features, 42, bias = True)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.86)
    model.to(device)

    model.load_state_dict(torch.load('./my_model.pth'))

    total = 0
    correct = 0

    with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (labels == predicted).sum().item()

    print('accuracy: ', 100*correct/total, '\n')