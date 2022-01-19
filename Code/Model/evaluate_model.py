import os
import sys
import torch
from pathlib import Path

#importing tensorflow and torch
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms.functional as F

#add the parent folder to the path so modules can be imported
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(parent_dir, "Helperfunctions"))
sys.path.append(os.path.join(parent_dir, "Preprocessing"))

#importing needed functions
import config
from slicedataset import Dataloading
from change_head import change_headsize
from to_image import convert_to_segmented_imgs


def Intersect(label_array, output_array, num_classes=4):
    intersect_per_class = []
    label_occurence_per_class = []
    output_occurence_per_class = []

    # Looping through the different classes.
    for c in range(num_classes):
        intersect = 0
        label_occurence = 0
        output_occurence = 0

        # Looping through the pixels in the output and the label and counting the intersected pixels and the occurence of the class in the images.
        for i in range(len(label_array)):
            if label_array[i] == c and output_array[i] == c:
                intersect += 1
                label_occurence += 1
                output_occurence += 1
            elif output_array[i] == c:
                output_occurence += 1
            elif label_array[i] == c:
                label_occurence += 1
        
        intersect_per_class.append(intersect)
        label_occurence_per_class.append(label_occurence)
        output_occurence_per_class.append(output_occurence)
    
    # Returning a tuple of 3 lists. Each list contains as much values as the number of classes.
    return (intersect_per_class, label_occurence_per_class, output_occurence_per_class)


def Dice(label_stack, output_stack, num_classes=4, bg_weight=0.25, smooth=1):
    # Sets weights for calculating the Dice / Change bg_weight to adjust the score.
    weights = [bg_weight, (1-bg_weight)/3, (1-bg_weight)/3, (1-bg_weight)/3]
    
    # Create a list of the stacked labels and a list of the stacked outputs as segmented images.
    label_list = []
    for k in range(label_stack.size(dim=0)):
        label_list.append(label_stack[k,:,:])
    output_list = convert_to_segmented_imgs(output_stack)
    
    # Looping through the different images of the stack.
    total_dice = 0
    for i in range(len(label_list)):
        print(f"<-- Sample: {i+1} -->")

        # Create flattened array of label and output.
        label_f = label_list[i].numpy().flatten().astype(int)
        output_f = output_list[i].flatten().astype(int)

        weighted_dice = 0

        # Use function Intersect to calculate the intersection and occurences per class.
        intersect_per_class, label_occurence_per_class, output_occurence_per_class = Intersect(label_f, output_f)

        # Calculate the dice scores for every class and adds them with their weights to the total dice of the image.
        for c in range(num_classes):
            dice_class = (2 * intersect_per_class[c] * smooth) / (label_occurence_per_class[c] + output_occurence_per_class[c] + smooth)
            weighted_dice += dice_class * weights[c]
 
        total_dice += weighted_dice
        print(f"Dice: {weighted_dice}\n")
    
    # Returns the sum of all the dice scores, this will be devided by the number of images later on.
    return total_dice


def training_model(test_size=0.2, num_epochs=10, batch_size=8, learning_rates=[0.001], pretrained=True, shuffle=True, array_path=config.array_dir):
    """Trains the model using the dataloader
    Args:
        test_size: fraction of data used for testing.
        num_epochs: number of epochs used for training.
        batch_size: size of the batches.
        learning_rates: list of learning rates.
        pretrained: True: use the pretrained model | False: use model without pretraining.
        shuffle: True: enables shuffle | False: disables shuffle
        array_path: path to the folder containing the arrayfiles per slice.
    """
    # Creating file to save dice scores.
    f = open("Dice_scores.txt", "w")

    # Loading datafiles.
    dataloading = Dataloading(test_size=test_size, array_path=array_path, batch_size=batch_size, shuffle=shuffle)
    
    # Creating a fcn model (this model is pretrained if pretrained is set to True).
    fcn = fcn_resnet50(pretrained=pretrained)

    # Setting model to trainingmode and set the device.
    fcn.train()
    device = "cuda"
    
    # Freezing its parameters.
    for param in fcn.parameters():
        param.requires_grad = False
    
    # Change head to output 4 classes.
    fcn = change_headsize(fcn, 4).to(device)
    
    # Find total parameters and trainable parameters.
    total_params = sum(p.numel() for p in fcn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in fcn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    
    # Looping through the different learning rates.
    for LR in learning_rates:
        # Looping through the number of epochs.
        for epoch in range(num_epochs):
            print(f"<----------Learning rate: {LR} --> Epoch: {epoch+1} ---------->")
            epoch_train_dice = 0.0
            epoch_eval_dice = 0.0
            fcn.train()
            # Model training loop
            print("Going through training data....")
            
            for i_batch, batch in enumerate(dataloading.train_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)

                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                optimizer.zero_grad()
                output = fcn(data)
                loss = criterion(output["out"], target.long())
                loss.backward()
                optimizer.step()
                
                epoch_train_dice += Dice(batch["label"], output["out"].detach())
                
            print("Going through testing data....")

            # Calculate validation loss after training
            fcn.eval()
            for i_batch, batch in enumerate(dataloading.test_dataloader):
                criterion = torch.nn.CrossEntropyLoss()
                data = F.convert_image_dtype(batch["image"], dtype=torch.float).to(device)
                target = batch["label"].to(device)
                output = fcn(data)

                epoch_eval_dice += Dice(batch["label"], output["out"].detach())
            

            train_dice = epoch_train_dice/len(dataloading.train_slicedata)
            eval_dice = epoch_eval_dice/len(dataloading.test_slicedata)
            f.write(f"____________ Epoch: {epoch+1} ____________\n")
            print("Training Dice:", train_dice)
            f.write(f"Training Dice: {train_dice} \n")
            print("Evaluation Dice:", eval_dice)
            f.write(f"Evaluation Dice: {eval_dice} \n")
            
        
        # Saving calculated weights in a separate file.
        torch.save(fcn.state_dict(), os.path.join(current_dir, f"weights_lr{str(int(LR*10000))}_e{str(num_epochs)}.h5"))
    
    f.close()

def running_model(pretrained=True, batch_size=8, loadingfile="weights.h5"):
    """Running the model and testing it on 1 sample
    Args:
        pretrained: True: use the pretrained model, False: use model without pretraining.
        batch_size: size of the batches. 
        loadingfile: filename of weightsfile.
    """
    # Creating FCN model and loading the weights from the loadingfile.
    fcn = fcn_resnet50(pretrained=pretrained)
    device = "cuda"
    fcn = change_headsize(fcn, 4)
    fcn.load_state_dict(torch.load(os.path.join(current_dir, loadingfile)))
    fcn.eval()

    # Creating dataloaders
    dataloading = Dataloading(test_size=0.2, array_path=config.array_dir, batch_size=batch_size, shuffle=True)
    total_dice = 0
    
    # Making predictions and calculating dice score
    print("Calculating dice score...")
    for i_batch, batch in enumerate(dataloading.test_dataloader):
        sample = batch["image"]
        sample = F.convert_image_dtype(sample, dtype=torch.float)
        output = fcn(sample)["out"]
        total_dice += Dice(batch["label"], output.detach())
    
    dice = total_dice/len(dataloading.test_slicedata)

    print(f"Overall dice score: {dice}")


def main():
    # To train the model and look at the dice scores during training, set 'train' to True.
    train = False
    
    # Set the training paramaters.
    learningrates = [0.001]
    num_epochs = 8
    batch_size = 16
    test_size = 0.2

    if train:
        training_model(learning_rate=learningrates, batch_size=batch_size, num_epochs=num_epochs, test_size=test_size)

    # If you want to run a model to evaluate testing data you can set 'run' to True and specify the name of the weights file. 
    # (Only works if you have trained the model and saved the weights)
    run = True
    file_to_load = "weights_lr10_e15.h5"

    if run:
        running_model(loadingfile=file_to_load, batch_size=batch_size)

    
if __name__ == '__main__':
    import timeit
    
    start = timeit.default_timer()

    main()

    stop = timeit.default_timer()

    print('Time: ', stop - start) 