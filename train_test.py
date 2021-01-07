'''
Author: Abhijit Adhikary
Email: u7035746@anu.edu.au
'''

from helper_functions import *
import torch
import torch.nn.functional as F

# function for testing the model
def test_model(test_parameters):

    model, device, test_loader, criterion = test_parameters
    # put model on evaluation mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        
        test_loss = 0
        test_correct = 0
        total = 0
        
        pred_array = []
        actual_array = []
        
        for current_batch in test_loader:

            images, labels_actual = current_batch
            images, labels_actual = images.to(device), labels_actual.to(device)
    
            labels_predicted = model.forward(images)
            loss = criterion(labels_predicted, labels_actual)
            test_loss += loss.item()

            labels_predicted = torch.argmax(F.softmax(labels_predicted, dim=1), dim=1)
            
            current_pred_array = labels_predicted.data.cpu().numpy()
            current_actual_array = labels_actual.data.cpu().numpy()
            test_correct += np.sum(current_pred_array == current_actual_array)

            total += labels_predicted.data.cpu().numpy().shape[0]
    
            pred_array.append(current_pred_array)
            actual_array.append(current_actual_array)
        
        test_accuracy = 100 * (test_correct / total)
        test_loss = test_loss / total
        
        num_classes = 7
        classsise_accuracy_array, precicion_array, recall_array, specificity_array = \
        get_SPI_calculations(pred_array, actual_array, num_classes, num_samples=total)
        
        SPI_parameters = \
        classsise_accuracy_array, precicion_array, recall_array, specificity_array, num_classes
        
        confusion_matrix = get_confusion_matrix(pred_array, actual_array, num_classes, total)
    
    return test_loss, test_accuracy, SPI_parameters, confusion_matrix

# function for validating the model
def validate_model(validation_parameters):
    
    model, device, validation_loader, criterion = validation_parameters
    model.to(device)
    
    with torch.no_grad():
        
        validation_loss = 0
        validation_correct = 0
        total = 0
        
        for current_batch in validation_loader:

            images, labels_actual = current_batch
            images, labels_actual = images.to(device), labels_actual.to(device)

            labels_predicted = model.forward(images)
            loss = criterion(labels_predicted, labels_actual)
            validation_loss += loss.item()

            labels_predicted = torch.argmax(F.softmax(labels_predicted, dim=1), dim=1)
            validation_correct += np.sum(labels_predicted.data.cpu().numpy() == labels_actual.data.cpu().numpy())
            total += labels_predicted.data.cpu().numpy().shape[0]
        validation_accuracy = 100 * (validation_correct / total)
        validation_loss = validation_loss / total
        
    return validation_loss, validation_accuracy

# function for training the model
def train_model(train_parameters):

    model, criterion, num_epochs, device, train_loader, validation_interval, validation_loader, \
        writer, save_checkpoint_condition,  model_name, epoch_start, learning_rate_scheduler, optimizer,\
        timestamp, comment, num_layers_to_freeze = train_parameters
    model.to(device)
    num_steps = 0

    for current_epoch in tqdm(range(epoch_start, num_epochs+1)):

        # put model to training mode
        # model.train()
       
        train_loss = 0
        train_correct = 0
        total = 0

        for current_batch in train_loader:

            num_steps += 1
            images, labels_actual = current_batch
            images, labels_actual = images.to(device), labels_actual.to(device)

            labels_predicted = model.forward(images)
            loss = criterion(labels_predicted, labels_actual)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # learning_rate_scheduler.step()
            
            labels_predicted = torch.argmax(F.softmax(labels_predicted, dim=1), dim=1)
            train_correct += np.sum(labels_predicted.data.cpu().numpy() == labels_actual.data.cpu().numpy())
            total += labels_predicted.data.cpu().numpy().shape[0]
        train_accuracy = 100 * (train_correct / total)
        train_loss = train_loss / total
            
        if num_steps % validation_interval == 0:
            # put the model to evaluation mode and validate
            model.eval()
            validation_parameters = model, device, validation_loader, criterion
            validation_loss, validation_accuracy = validate_model(validation_parameters)
            print_to_console = False
            if print_to_console:
                print(f'Epoch: {current_epoch}/{num_epochs}', end='\t')
                print(f'T.Loss: {train_loss:.6f}', end='\t')
                print(f'T.Accuracy: {train_accuracy:.2f} %', end='\t')

                print(f'V.Loss: {validation_loss:.6f}', end='\t')
                print(f'V.Accuracy: {validation_accuracy:.2f} %')

            # save the current training statistics for visualizing in TensorBoard
            writer.add_scalar('Loss/Train', train_loss, current_epoch)
            writer.add_scalar('Loss/Validation', validation_loss, current_epoch)
            
            writer.add_scalar('Accuracy/Train', train_accuracy, current_epoch)
            writer.add_scalar('Accuracy/Validation', validation_accuracy, current_epoch)

            writer.add_scalars('Loss', {'Loss/Train':train_loss, \
                                    'Loss/Validation': validation_loss}, current_epoch)
            writer.add_scalars('Accuracy', {'Accuracy/Train':train_accuracy, \
                                    'Accuracy/Validation': validation_accuracy}, current_epoch)
        # save the model checkpoint at the last train epoch
        if (current_epoch == num_epochs) and save_checkpoint_condition:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            checkpoint_parameters = model, model_name, current_epoch, num_epochs, timestamp, comment, num_layers_to_freeze
            save_checkpoint(checkpoint_parameters)
    torch.cuda.empty_cache()