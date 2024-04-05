import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
import sys
from tqdm import tqdm
from metrics import *
from loss import BCEDiceLoss
import pandas as pd
import matplotlib.pyplot as plt
# from torchsummary import summary

def train(FLAGS):

    # Defining the hyperparameters
    device =  FLAGS.cuda
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate
    save_every = FLAGS.save_every
    nc = FLAGS.num_classes
    wd = FLAGS.weight_decay
    ip = FLAGS.input_path_train
    lp = FLAGS.label_path_train
    ipv = FLAGS.input_path_val
    lpv = FLAGS.label_path_val
    print ('[INFO]Defined all the hyperparameters successfully!')
    
    # Get the class weights
    print ('[INFO]Starting to define the class weights...')
    pipe = loader(ip, lp, batch_size='all')
    class_weights = get_class_weights(pipe, nc)
    print ('[INFO]Fetched all class weights successfully!')

    # Get an instance of the model
    enet = ENet(nc)
    print ('[INFO]Model Instantiated!')
    
    # Move the model to cuda if available
    enet = enet.to(device)

    # Print model summary
    # summary(enet, (3, 512, 512))

    # Print model architecture
    print('[INFO] Model Architecture:')
    print(enet)

    # Define the criterion and the optimizer
    # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    criterion = nn.CrossEntropyLoss()
    # criterion = BCEDiceLoss(weight_dice=2, smooth=0.1, class_weights=torch.FloatTensor(class_weights).to(device))
    # criterion = BCEDiceLoss(weight_dice=2, smooth=0.1)
    optimizer = torch.optim.Adam(enet.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    print ('[INFO]Defined the loss function and the optimizer')

    # Training Loop starts
    print ('[INFO]Starting Training...')
    print ()

    # Data storage lists
    train_data = []
    eval_data = []

    train_losses = []
    eval_losses = []
    train_accuracies = []
    eval_accuracies = []
    train_miou_scores = []
    eval_miou_scores = []
    train_f1_scores = []
    eval_f1_scores = []
    # train_precisions = []
    # eval_precisions = []
    # train_recalls = []
    # eval_recalls = []

    # Assuming we are using the CamVid Dataset
    bc_train = 721 // batch_size
    bc_eval = 206 // batch_size

    pipe = loader(ip, lp, batch_size)
    eval_pipe = loader(ipv, lpv, batch_size)

    epochs = epochs
            
    for e in range(1, epochs+1):
            
        train_loss = 0
        train_accuracy = 0
        train_miou = 0
        train_precision = 0
        train_recall = 0
        train_f1 = 0

        print ('-'*15,'Epoch %d' % e, '-'*15)
        
        enet.train()
        
        for _ in tqdm(range(bc_train)):
            X_batch, mask_batch = next(pipe)
            
            #assert (X_batch >= 0. and X_batch <= 1.0).all()
            
            X_batch, mask_batch = X_batch.to(device), mask_batch.to(device)

            optimizer.zero_grad()

            out = enet(X_batch.float())
            # print("out.shape", out.shape, "mask_batch.shape", mask_batch.shape)

            # mask_batch = mask_batch.to(torch.float32) # new code line added to convert label to float data type. Comment this when using CE loss
            # loss = criterion(out, mask_batch) # use this when using BCEDiceLoss. Comment this when using CE loss
            loss = criterion(out, mask_batch.long()) # Uncomment this for CE loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            """
            # Calculate training performance metrics
            train_accuracy = calculate_accuracy(out, mask_batch)
            train_confusion_matrix = calculate_confusion_matrix(out, mask_batch, nc)
            train_miou = calculate_miou(train_confusion_matrix)
            train_f1, train_precision, train_recall = calculate_f1_score(train_confusion_matrix)
        
        print('Train Loss: {:.6f}'.format(loss.item()),
              'Train Accuracy: {:.6f}'.format(train_accuracy),
              'Train mIoU: {:.6f}'.format(train_miou),
              'Train F1 Score: {:.6f}'.format(train_f1),
              'Train Precision: {:.6f}'.format(train_precision.mean().item()),
              'Train Recall: {:.6f}'.format(train_recall.mean().item()))

        train_data.append([e, train_loss, train_accuracy, train_miou, train_f1, train_precision.mean().item(), train_recall.mean().item()])
        """

        train_losses.append(train_loss)

        """
        train_accuracies.append(train_accuracy)
        train_miou_scores.append(train_miou)
        train_f1_scores.append(train_f1)
        # train_precisions.append(train_precision.mean().item())
        # train_recalls.append(train_recall.mean().item())
        """

        with torch.no_grad():
            enet.eval()
                
            eval_loss = 0
                
            for _ in tqdm(range(bc_eval)):
                inputs, labels = next(eval_pipe)

                inputs, labels = inputs.to(device), labels.to(device)

                # inputs = inputs.to(torch.float32) # new code line added to convert inputs to float. Comment this for CE loss
                out = enet(inputs)

                # labels = labels.to(torch.float32) # new code line added to convert labels to float. Comment this for CE loss                    
                # loss = criterion(out, labels) # Use this for BCEDice loss. Comment for CE loss
                loss = criterion(out, labels.long()) # Uncomment this for CE loss

                eval_loss += loss.item()

                """
                # Calculate validation performance metrics
                eval_accuracy = calculate_accuracy(out, labels)
                eval_confusion_matrix = calculate_confusion_matrix(out, labels, nc)
                eval_miou = calculate_miou(eval_confusion_matrix)
                eval_f1, eval_precision, eval_recall = calculate_f1_score(eval_confusion_matrix)

            print('Valid Loss: {:.6f}'.format(loss.item()),
                  'Valid Accuracy: {:.6f}'.format(eval_accuracy),
                  'Valid mIoU: {:.6f}'.format(eval_miou),
                  'Valid F1 Score: {:.6f}'.format(eval_f1),
                  'Valid Precision: {:.6f}'.format(eval_precision.mean().item()),
                  'Valid Recall: {:.6f}'.format(eval_recall.mean().item()))

            eval_data.append([e, eval_loss, eval_accuracy, eval_miou, eval_f1, eval_precision.mean().item(), eval_recall.mean().item()])
            """

            eval_losses.append(eval_loss)

            """
            eval_accuracies.append(eval_accuracy)
            eval_miou_scores.append(eval_miou)
            eval_f1_scores.append(eval_f1)
            # eval_precisions.append(eval_precision.mean().item())
            # eval_recalls.append(eval_recall.mean().item())
            """            

        if e % save_every == 0:
            checkpoint = {
                'epochs' : e,
                'state_dict' : enet.state_dict()
            }
            torch.save(checkpoint, './ckpt-enet-{}-{}.pth'.format(e, train_loss))
            print ('Model saved!')

    """
    # Create DataFrames
    train_df = pd.DataFrame(train_data, columns=['Epoch', 'Train Loss', 'Train Accuracy', 'Train mIoU', 'Train F1 Score', 'Train Precision', 'Train Recall'])
    eval_df = pd.DataFrame(eval_data, columns=['Epoch', 'Valid Loss', 'Valid Accuracy', 'Valid mIoU', 'Valid F1 Score', 'Valid Precision', 'Valid Recall'])

    # Save to Excel
    train_df.to_csv('train_metrics.csv', index=False)
    eval_df.to_csv('eval_metrics.csv', index=False)

    """
    # Plotting
    plt.figure(figsize=(12, 8))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), eval_losses, label='Valid Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    """
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), eval_accuracies, label='Valid Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # mIoU plot
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), train_miou_scores, label='Train mIoU')
    plt.plot(range(1, epochs+1), eval_miou_scores, label='Valid mIoU')
    plt.title('Training and Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()

    # F1 score plot
    plt.subplot(2, 2, 4)
    plt.plot(range(1, epochs+1), train_f1_scores, label='Train F1 Score')
    plt.plot(range(1, epochs+1), eval_f1_scores, label='Valid F1 Score')
    plt.title('Training and Validation F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    """

    plt.tight_layout()

    # Save plots
    plt.savefig('training_plots.png')
    plt.show()

    # Print total mean loss after all epochs
    print('Epoch {}/{}...'.format(e+1, epochs))
    print('[INFO] Total Mean Train Loss: {:.6f}'.format(sum(train_losses) / len(train_losses)))
    print('[INFO] Total Mean Valid Loss: {:.6f}'.format(sum(eval_losses) / len(eval_losses)))
    print('[INFO] Training Process complete!')

    # Function to save segmentation masks as images
    def save_segmentation_masks(model, dataloader, save_dir, device):
        model.eval()
        with torch.no_grad():
            for i, (inputs, _) in enumerate(dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                predictions = predictions.cpu().numpy()
            
                for j in range(len(predictions)):
                    prediction = predictions[j]
                    prediction_img = Image.fromarray(prediction.astype(np.uint8))
                    prediction_img.save(os.path.join(save_dir, f'output_{i * dataloader.batch_size + j}.png'))

    output_save_dir = 'output_images'
    os.makedirs(output_save_dir, exist_ok=True)
    save_segmentation_masks(enet, eval_pipe, output_save_dir, device)
    print('Segmentation masks saved as images!')
