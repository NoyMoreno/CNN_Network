from torch.autograd import Variable
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define classes
classes = ('bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four',
           'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on',
           'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
           'up', 'wow', 'yes', 'zero')
# Hyper-parameters
num_epochs = 20
batch_size = 10
learning_rate = 0.01

# train set
dataset = GCommandLoader('/content/train')
# train loader
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,
    pin_memory=True)
# validation set
validation_set = GCommandLoader('/content/valid')
# validation loader
validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=batch_size, shuffle=True,
    pin_memory=True)
# test set
test_set = GCommandLoader('/content/test_')
# test loader
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False,
    pin_memory=True)


def train():
    print("start training...")
    model.train()
    train_loss = 0
    correct = 0
    # iterate once over training_loader (1 epoc)
    for batch_idx, (train_x, target) in enumerate(train_loader):
        train_x = train_x.cuda()
        target = target.cuda()
        train_x, target = Variable(train_x), Variable(target)
        optimizer.zero_grad()
        output = model(train_x)
        output = output.cuda()
        loss = F.nll_loss(output, target)
        loss = loss.cuda()
        loss.backward()
        optimizer.step()
        # calculate loss and accuracy for report file
        train_loss += loss
        train_loss = train_loss.cuda()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()
        correct=correct.cuda()
    train_loss /= len(train_loader.dataset) / batch_size
    print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def validation_test():
    print("start Validation test..")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(validation_loader):
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            output= output.cuda()
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            pred = pred.cuda()
            correct += pred.eq(target.view_as(pred)).cpu().sum()
            correct=correct.cuda()
    test_loss /= len(validation_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


def test_predications():
    # Get file names
    file_names = test_loader.sampler.data_source.file_names
    file_names.remove('.DS_Store')
    i = 0
    files_to_predications = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            files_to_predications.append(file_names[i] + ',' + classes[predicted.item()])
            i += 1
    # Sort by file index
    files_to_predications = sorted(files_to_predications, key=lambda x: int(x.split('.')[0]))
    # Write all predictions to file
    with open('test_y', 'w') as out:
        for i in range(len(files_to_predications)):
            out.write(files_to_predications[i])
            # In the last prediction do not add a blank line
            if not i == len(files_to_predications) - 1:
                out.write('\n')
    print("Finish test predications")


if __name__ == '__main__':
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # drive
    for epoch in range(num_epochs):
        print(epoch)
        train()
        validation_test()
    # Finish training, let's predict the test
    test_predications()
