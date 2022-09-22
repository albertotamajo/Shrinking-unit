# Built-in modules
from shrinkingunit import Classifier, ShrinkingUnitStack

# Third-party modules
import torch.optim
from torch_geometric.datasets import ModelNet
import torch_geometric.data as data
from torch_geometric.transforms import SamplePoints, NormalizeScale, NormalizeRotation
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import os, re
import statistics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import shutil
import random
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Optional, Sized
from torch.utils.data.distributed import Sampler
import torch
import math

torch.set_printoptions(threshold=10_000)
torch.autograd.set_detect_anomaly(True)

# seed for reproducability
seed_no = 6 #random.randint(0, 1000)
random.seed(seed_no)

# Hyperparameters
optimizer_str = "Adam"
sample_points = 1200
learning_rate = 3e-4
weight_decay = 0
momentum = 0.999
batch_size = 8
epochs = 200
dimensionality = 3 # 3 for x,y,z or 6 for x,y,z,nx,ny,nz
out_points_decay = 5
patience = 100 #early stopping patience
data_shuffle = False
P_values = [3, 6, 6] #for point cloud feature augmentation

# others
classes = {0:'Bathtub', 1:'Bed', 2:'Chair', 3:'Desk', 4:'Dresser', 5:'Monitor', 6:'Night Stand', 7:'Sofa', 8:'Table', 9:'Toilet'}
overall_classes_loss = False #True for weights over all classes, False for batchwise weighting
verbosity = True # for output level detail, True for maximum, False for minimum output information


class MLP2Layers(nn.Module):
    def __init__(self, in_feature1, out_feature1, out_feature2, out_feature3, out_feature4, out_feature5, out_feature6):
        super().__init__()
        lin1 = nn.Linear(in_feature1, out_feature1)
        lin2 = nn.Linear(out_feature1, out_feature2)
        lin3 = nn.Linear(out_feature2, out_feature3)
        lin4 = nn.Linear(out_feature3, out_feature4)
        lin5 = nn.Linear(out_feature4, out_feature5)
        lin6 = nn.Linear(out_feature5, out_feature6)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        nn.init.xavier_uniform_(lin6.weight)
        self.neuralNet = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.Tanh(),
            lin4,
            nn.ReLU(),
            lin5,
            nn.ReLU(),
            lin6,
            nn.ReLU()
        )

    def forward(self, X):
        return self.neuralNet(X)


# ----------------------------------------------------------BEGINNING MODEL--------------------------------------------#
# First Shrinking Layer
input_feature = dimensionality # keep that node fix
out_points = sample_points // out_points_decay
input_stack = 1
stack_fork = 2
n_init = 1
C = input_feature
P = P_values[0]
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayerStart = ShrinkingUnitStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                         M, C, P, mlp1, mlp2)

# Shrinking Layer 2
out_points = out_points // out_points_decay
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 3
C = input_feature
P = P_values[1]
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayer2 = ShrinkingUnitStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                     M, C, P, mlp1, mlp2)

# Last Shrinking Layer
out_points = 1 #keep that node fix
n_init = 1
input_stack = input_stack * stack_fork
stack_fork = 2
C = input_feature
P = P_values[2]
out_feature = C + P
mlp = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C)
sigma = nn.ReLU()
F = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
W = MLP2Layers(C, C + 10, C + 15, C + 20, C + 15, C + 10, C * (C + P))
M = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, 1)
mlp1 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
mlp2 = MLP2Layers(C + P, C + P + 10, C + P + 15, C + P + 20, C + 15, C + P + 10, C + P)
input_feature = out_feature
shrinkingLayerEnd = ShrinkingUnitStack(input_stack, stack_fork, mlp, 1.0, out_points, "k-means++", n_init, sigma, F, W,
                                       M, C, P, mlp1, mlp2)

# Shrinking layers
shrinkingLayers = [shrinkingLayerStart, shrinkingLayer2, shrinkingLayerEnd]


# MLP classifier
class MLPClassifer(nn.Module):
    def __init__(self, in_feature: int):
        super().__init__()
        lin1 = nn.Linear(in_feature, in_feature * 2)
        lin2 = nn.Linear(in_feature * 2, in_feature * 2 + 10)
        lin3 = nn.Linear(in_feature * 2 + 10, in_feature * 2 + 20)
        lin4 = nn.Linear(in_feature * 2 + 20, in_feature * 2 + 10)
        lin5 = nn.Linear(in_feature * 2 + 10, 10)
        nn.init.xavier_uniform_(lin1.weight)
        nn.init.xavier_uniform_(lin2.weight)
        nn.init.xavier_uniform_(lin3.weight)
        nn.init.xavier_uniform_(lin4.weight)
        nn.init.xavier_uniform_(lin5.weight)
        self.main = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            lin3,
            nn.ReLU(),
            lin4,
            nn.ReLU(),
            lin5
        )

    def forward(self, feature_matrix_batch):
        output = self.main(feature_matrix_batch.squeeze())
        return output

mlpClassifier = MLPClassifer(input_feature)


# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #

def save_checkpoint(model, optimizer, scheduler, epoch, epoch_losses, training_accuracies, test_losses, test_accuracies, learning_rates, save_path, train_model_results, test_model_results, train_conf_mat, test_conf_mat):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_losses': epoch_losses,
        'training_accuracies': training_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates,
        'batchwise_training_results': train_model_results,
        'batchwise_test_results': test_model_results,
        'training_conf_mat': train_conf_mat,
        'test_conf_mat': test_conf_mat
    }, save_path)


def load_checkpoint(model, optimizer, scheduler, load_path):
    try:
        checkpoint = torch.load(load_path)
        print("Progress file in the folder")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model state diction read")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state dictionary read")
        epoch = checkpoint['epoch']
        print("Epoch read")
        print(epoch + 1)
        return checkpoint
    except:
        print("Progress file not in the folder")
        return 0


def printLearningRate(optimizer):
    for param_group in optimizer.param_groups:
        print(f"\nLearning rate: {param_group['lr']}")


# Training loop
def training_loop(gpu, training_dataloader, model, loss_fn, optimizer):
    losses = []
    correct = 0
    batch_results = dict()
    conf_mat = np.zeros((10,10))
    for batch_n, batch in enumerate(training_dataloader): #batch[batch, pos, ptr, y]
        batch_size = int(batch.batch.size()[0] / sample_points)        
        
        if dimensionality == 3:
            # Input dim [:,3] for your geometry x,y,z
            X = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1) + torch.normal(
                torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points,
                                                                                   dimensionality), fill_value=0.1)).cuda(gpu)
        else:
            # Input dim [:,6] for your geometry x,y,z and normals nx,ny,nz
            X = torch.cat((batch.pos.cuda(non_blocking=True), batch.normal.cuda(non_blocking=True)), 1).view(batch_size, sample_points, -1) + torch.normal(
                torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points,
                                                                                   dimensionality), fill_value=0.1)).cuda(gpu)
        
        y = batch.y.cuda(non_blocking=True).flatten() #size (batch_size) --> torch.Size([8])
        
        # Compute predictions
        pred = model(None, X) #size (batch_size,classes) --> torch.Size([8, 10])
        
        if overall_classes_loss:
            # weighted CE Loss over all classes
            loss = loss_fn(pred, y)
        else:
            # weighted batchwise Loss
            sample_count = np.array([[x, batch.y.tolist().count(x)] for x in batch.y])[:,1]
            batch_weights = 1. / sample_count
            batch_weights = torch.from_numpy(batch_weights)
            batch_weights = batch_weights.double()
            loss = element_weighted_loss(pred, batch.y, batch_weights, gpu)
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        print(f"Loss: {loss}")

        tensor_list_y =  [torch.ones_like(y) for _ in range(dist.get_world_size())]
        tensor_list_pred = [torch.ones_like(y) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(tensor_list_y, y, group=None, async_op=False)
        torch.distributed.all_gather(tensor_list_pred, pred.argmax(1), group=None, async_op=False)
        tensor_list_y = torch.cat(tensor_list_y)
        tensor_list_pred = torch.cat(tensor_list_pred)
        
        # Confusion Matrix
        conf_mat += confusion_matrix(tensor_list_y.cpu().detach().numpy(), tensor_list_pred.cpu().detach().numpy(), labels=np.arange(0,10))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Save batch predictions
        batch_results[batch_n] = {'true':tensor_list_y, 'pred':tensor_list_pred}
        
        if verbosity == True:
            print(f"\n\nTRAIN on GPU:{gpu}: True Label {y} - Prediction {pred.argmax(1)} - Loss {loss}")
            truevalue = '\t\t'.join(classes[items] for items in y.tolist())
            predvalues = '\t\t'.join(classes[items] for items in pred.argmax(1).tolist())
            print(f"INFO on GPU:{gpu}: TRAIN - True Value\t {truevalue}")
            print(f"INFO on GPU:{gpu}: TRAIN - Predictions\t {predvalues}")
        
        if batch_n % 25 == 0:
            torch.distributed.reduce(loss, 0)
            """if gpu == 0:
                # print predictions and true values
                #truevalue = '\t\t'.join(classes[items] for items in y.tolist())
                #predvalues = '\t\t'.join(classes[items] for items in pred.argmax(1).tolist())
                #print(f"\n\nINFO on GPU{gpu}: TRAIN - True Value\t {truevalue}")
                #print(f"INFO on GPU{gpu}: TRAIN - Predictions\t {predvalues}")
                #print("INFO: TRAIN - Correctness\t", pred.argmax(1) == y)
                #print(f"INFO: TRAIN - Single Batch Test Accuracy {correct * 100 / batch_size}\n\n")
                loss, current = loss.item(), batch_n * len(X)
                #print(f"loss: {loss:>7f}")"""
        #print(f"conf_mat: {conf_mat}")
        #print(f"batch_results: {batch_results}")

    return torch.tensor(losses, device=f"cuda:{gpu}"), torch.tensor(correct, device=f"cuda:{gpu}"), batch_results, conf_mat

# Test loop
def test_loop(gpu, test_dataloader, model, loss_fn):
    test_losses = []
    correct = 0
    batch_results = dict()
    conf_mat = np.zeros((10,10))
    with torch.no_grad():
        for batch_n, batch in enumerate(test_dataloader):
            batch_size = int(batch.batch.size()[0] / sample_points)
            
            if dimensionality == 3:
                # Input dim [:,3] for your geometry x,y,z
                X = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1)
            else:
                # Input dim [:,6] for your geometry x,y,z and normals nx,ny,nz
                X = torch.cat((batch.pos.cuda(non_blocking=True), batch.normal.cuda(non_blocking=True)), 1).view(batch_size, sample_points, -1)
                
            y = batch.y.cuda(non_blocking=True).flatten()
            pred = model(None, X) #size (batch,classes) per batch_n
            
            if overall_classes_loss:
                # weighted CE Loss over all classes
                loss = loss_fn(pred, y)
            else:
                # weighted batchwise Loss
                sample_count = np.array([[x, batch.y.tolist().count(x)] for x in batch.y])[:,1]
                batch_weights = 1. / sample_count
                batch_weights = torch.from_numpy(batch_weights)
                batch_weights = batch_weights.double()
                loss = element_weighted_loss(pred, batch.y, batch_weights, gpu)
            
            test_losses.append(loss.item())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f"Loss: {loss}")
        
            tensor_list_y =  [torch.ones_like(y) for _ in range(dist.get_world_size())]
            tensor_list_pred = [torch.ones_like(y) for _ in range(dist.get_world_size())]
            torch.distributed.all_gather(tensor_list_y, y, group=None, async_op=False)
            torch.distributed.all_gather(tensor_list_pred, pred.argmax(1), group=None, async_op=False)
            tensor_list_y = torch.cat(tensor_list_y)
            tensor_list_pred = torch.cat(tensor_list_pred)
            
            # Confusion Matrix
            conf_mat += confusion_matrix(tensor_list_y.cpu().detach().numpy(), tensor_list_pred.cpu().detach().numpy(), labels=np.arange(0,10))
            
            # Save batch predictions
            batch_results[batch_n] = {'true':tensor_list_y, 'pred':tensor_list_pred}
            
            if verbosity == True:
                print(f"\n\nTEST on GPU:{gpu}: True Label {y} - Prediction {pred.argmax(1)} - Loss {loss}")
                truevalue = '\t\t'.join(classes[items] for items in y.tolist())
                predvalues = '\t\t'.join(classes[items] for items in pred.argmax(1).tolist())
                print(f"INFO on GPU:{gpu}: TEST - True Value\t {truevalue}")
                print(f"INFO on GPU:{gpu}: TEST - Predictions\t {predvalues}")

    test_loss = statistics.mean(test_losses)
    return torch.tensor(correct, device=f"cuda:{gpu}"), torch.tensor(test_loss, device=f"cuda:{gpu}"), batch_results, conf_mat


def createdir(path):
    try:
        os.mkdir(path)
        print(f"\nINFO:Directory '{path}' created")
    except FileExistsError:
        print(f"INFO:Directory '{path}' already exists")


def next_training_number(path):
    listdir = os.listdir(path)
    if listdir == []:
        return 1
    else:
        list_number = map(lambda x: int(x.replace("train", "")), filter(lambda x: x.startswith("train"), listdir))
        return max(list_number) + 1 if list_number is not [] else 1


def early_stopping(test_losses):
    if len(test_losses) <= 1:
        return False
    else:
        if test_losses[-1] > test_losses[-2]:
            return True
        else:
            return False


def save_hyperparameters(save_path, file_name):
    with open(os.path.join(save_path, file_name), 'w') as f:
        f.write('Hyper-parameter configuration for train run ' + re.findall(r'[\d]+', save_path.split(os.sep)[-1])[0])
        f.write('\nOptimizer: ' + str(optimizer_str))
        f.write('\nNum of sample points: ' + str(sample_points))
        f.write('\nDimensionality: ' + str(dimensionality))
        f.write('\nBatch size: ' + str(batch_size) + ' -> (' + str((batch_size * int(gpus))) + ')')
        f.write('\nInitial learning_rate: ' + str(learning_rate))
        f.write('\nRegularization (weight_decay): ' + str(weight_decay))
        if optimizer_str != "Adam":
            f.write('\nMomentum: ' + str(momentum))
        if args.resume:
            final_epoch = (torch.load("stackgraphConvPool3DPnet/" + args.resume)['epoch'])
            f.write('\nInitial epoch to start retraining: ' + str(final_epoch))
            f.write('\nNum of epochs: ' + str(epochs))
        else:
            f.write('\nNum of epochs: ' + str(epochs))
        f.write('\nSeed: ' + str(seed_no))
        f.write('\nP-Values for ' + str(len(P_values)) + ' stacked Layers of size: ' + str(P_values))


def export_sampled_data(data, export_path):
    dict_labeled_3D = dict()
    for batch_n, batch in enumerate(data):
        for i in range(len(batch.y)):
            dict_labeled_3D[batch_n] = {'y':batch.y[i], 'sample_points':batch.pos[batch.ptr[i]:batch.ptr[i+1],:]}
    torch.save(dict_labeled_3D, export_path)


def element_weighted_loss(pred, y, weights, gpu):
    # batch wise weightes Loss
    y = y.cuda(gpu)
    weights = weights.cuda(gpu)
    m = torch.nn.LogSoftmax(dim=1).cuda(gpu)
    criterion = torch.nn.NLLLoss(reduction='none').cuda(gpu) #negative log likelihood loss
    loss = criterion(m(pred), y)
    loss = loss * weights
    return loss.sum() / weights.sum()


def train_optimisation(gpu, gpus, training_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, dir_path, initial_epoch):
    epoch_losses = []
    training_accuracies = []
    test_losses = []
    test_accuracies = []
    learning_rates = []
    counter = 0 #early stopping counter
    batchwise_results = dict()
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20)

    for i in range(initial_epoch, initial_epoch + epochs):
        if gpu == 0:
            if initial_epoch > 0:
                print(f"\n\nEpoch {i}\n-------------------------------")
            else:
                print(f"\n\nEpoch {i + 1}\n-------------------------------")

        # TRAIN
        losses, training_accuracy, train_batch_result, train_conf_mat = training_loop(gpu, training_dataloader, model, loss_fn, optimizer)
        average_loss = torch.mean(losses)
        torch.distributed.reduce(average_loss, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(training_accuracy, 0, torch.distributed.ReduceOp.SUM)
        
        # TEST
        test_accuracy, test_loss, test_batch_result, test_conf_mat = test_loop(gpu, test_dataloader, model, loss_fn)
        torch.distributed.reduce(test_accuracy, 0, torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(test_loss, 0, torch.distributed.ReduceOp.SUM)
                
        # save results
        batchwise_results[i] = {'train':train_batch_result, 'test':test_batch_result}
        if gpu == 0:  # the following operations are performed only by the process running in the first gpu
            average_loss = average_loss / torch.tensor(gpus, dtype=torch.float)  # average loss among all gpus
            test_accuracy = test_accuracy / torch.tensor(len(test_dataloader.dataset),
                                                         dtype=torch.float) * torch.tensor(100.0)
            training_accuracy = training_accuracy / torch.tensor(len(training_dataloader.dataset),
                                                                 dtype=torch.float) * torch.tensor(100.0)
            test_loss = test_loss / torch.tensor(gpus, dtype=torch.float)
            epoch_losses.append(average_loss.item())
            training_accuracies.append(training_accuracy.item())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_accuracy.item())
            learning_rates.append((optimizer.param_groups[0])["lr"])
            print(f"\nBatch size: {batch_size * int(gpus)}")
            print(f"average Training Loss: {average_loss.item():.6f}")
            print(f"average Test Loss: {test_loss.item():.6f}")
            print(f"\naverage Training Acc: {training_accuracy.item():.6f}")
            print(f"average Test Acc: {test_accuracy.item():.6f}")
            printLearningRate(optimizer)
            scheduler.step(test_loss)
            
            """# stepwise learning rate decay
            if average_loss.item() <= 0.35:
                for param_group in optimizer.param_groups:
                    print("Learning rate changed to 0.007")
                    param_group['lr'] = 0.007
            if average_loss.item() <= 0.30:
                for param_group in optimizer.param_groups:
                    print("Learning rate changed to 0.004")
                    param_group['lr'] = 0.004"""
            
            # saving model checkpoint
            save_checkpoint(model, optimizer, scheduler, i, epoch_losses, training_accuracies, test_losses, test_accuracies, learning_rates,
                            os.path.join(dir_path, f"epoch{i}.pth"), {key: value for key, value in batchwise_results[i].items() if key == 'train'}, {key: value for key, value in batchwise_results[i].items() if key == 'test'}, train_conf_mat, test_conf_mat)
            #TODO: implement ONNX Export
            # early stopping scheduler
            if early_stopping(test_losses) == True:
                counter += 1
                print(f"Early Stopping counter: {counter} of {patience}")
            else:
                counter += 0                        
            if counter < patience:
                pass
            else:
                print("\n\nEarly Stopping activated")
                print(f"Training stopped at Epoch{i + 1}")
                dist.destroy_process_group()
                exit()

# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #

# ---------------------------------------------CUSTOM SAMPLER--------------------------------------------------------- #

class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, data_source: Optional[Sized], num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0, replacement: bool = True):
        super().__init__(data_source)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement #sample can be drown again in that row if True

    def calculate_weights(self, targets):
        class_sample_count = np.array([len(np.where(self.dataset.data.y == t)[0]) for t in np.unique(self.dataset.data.y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in self.dataset.data.y])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        return samples_weigth

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # data.data.y == labels
        targets = self.dataset.data.y
        targets = targets[self.rank:self.total_size:self.num_replicas]
        #assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)
        weighted_indices = torch.multinomial(weights, self.num_samples, self.replacement).tolist()

        return iter(weighted_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

# ---------------------------------------------CUSTOM SAMPLER--------------------------------------------------------- #


# -------------------------------------------MULTI-GPU MODEL---------------------------------------------------------- #

def train(gpu, gpus, world_size):
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    try:
        dist.init_process_group(backend='nccl', world_size=world_size, rank=gpu) #for distributed GPU training
    except RuntimeError:
        print("\n\nINFO:RuntimeError is raised >> Used gloo backend instead of nccl!\n")
        dist.init_process_group(backend='gloo', world_size=world_size, rank=gpu) #as a fallback option
    
    dir_path = None
    if gpu == 0:
        dir_path = "stackgraphConvPool3DPnet"
        createdir(dir_path)
        training_number = next_training_number(dir_path)
        dir_path = os.path.join(dir_path, f"train{training_number}")
        createdir(dir_path)
        #save hyper-parameters in txt protocol file
        save_hyperparameters(dir_path, 'hyperparameters.txt')
        print("\nINFO: Protocol File saved successfully . . .")
        
        #copy crucial py-files in current train folder
        shutil.copy2(os.path.basename(__file__), dir_path)
        shutil.copy2('stackGraphConvPool3DPnet.py', dir_path)
        shutil.copy2('shrinkingunit.py', dir_path)
        shutil.copy2('utilities.py', dir_path)
        print("\nINFO: Script Files copied successfully . . .")

    model = Classifier(shrinkingLayers, mlpClassifier)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    #setting up optimizer
    if optimizer_str == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_str == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    
    # single-program multiple-data training paradigm (Distributed Data-Parallel Training)
    model = DDP(model, device_ids=[gpu])
    
    if dimensionality == 3:
        training_data = ModelNet("ModelNet10_train_data", transform=lambda x: NormalizeScale()(SamplePoints(num=sample_points)(x)))
    else:
        training_data = ModelNet("ModelNet10_train_data", transform=lambda x: NormalizeScale()(NormalizeRotation()(SamplePoints(num=sample_points, remove_faces=True, include_normals=True)(x))))
    
    training_sampler = DistributedWeightedSampler(training_data, num_replicas=world_size) #weight unbalanced classes by 1/cls_count
    training_dataloader = data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=data_shuffle, num_workers=0,
                                          pin_memory=True, sampler=training_sampler)
    
    if dimensionality == 3:
        test_data = ModelNet("ModelNet10_test_data", train=False, transform=lambda x: NormalizeScale()(SamplePoints(num=sample_points)(x)))
    else:
        test_data = ModelNet("ModelNet10_test_data", train=False, transform=lambda x: NormalizeScale()(NormalizeRotation()(SamplePoints(num=sample_points, remove_faces=True, include_normals=True)(x))))
    
    test_sampler = DistributedWeightedSampler(test_data, num_replicas=world_size) #weight unbalanced classes by 1/cls_count
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=data_shuffle, num_workers=0,
                                      pin_memory=True, sampler=test_sampler)
    
    """# save sampled data for later result visualisation
    try:
        #export_path = os.path.join("stackgraphConvPool3DPnet", "train" + str(next_training_number("stackgraphConvPool3DPnet")-1))
        #export_sampled_data(training_dataloader, os.path.join(export_path, "train_sampledPoints.pth"))
        #export_sampled_data(test_dataloader, os.path.join(export_path, "test_sampledPoints.pth"))
        print("\nINFO: Sampled 3D data successfully saved . . .")
    except Exception as e:
        print(f"\nERROR: Sampled 3D data could not saved successfully . . . - this process does not executed - caused by {e}")"""
    
    # weighted CE Loss over all Classes C
    class_sample_count = np.array([len(np.where(training_data.data.y == t)[0]) for t in np.unique(training_data.data.y)])
    weight = 1. / class_sample_count
    weight = torch.from_numpy(weight)
    weight = weight.float()
    loss_fn = nn.CrossEntropyLoss(weight=weight).cuda(gpu)
    
    # continue training from certain checkpoint
    continue_from_scratch = True if args.resume is None else False
    if continue_from_scratch: 
        if gpu == 0:
            print("\nINFO: Train from scratch has started . . .")
        train_optimisation(gpu, gpus, training_dataloader, test_dataloader, model, loss_fn, optimizer, None, dir_path, 0)
    else:
        checkpoint_path = "stackgraphConvPool3DPnet/" + args.resume
        if gpu == 0:
            print(f"\nINFO: Train has started from certain checkpoint {checkpoint_path.split('/')[2].split('.')[0]} in {checkpoint_path.split('/')[1]} . . .")
        model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)
        optimizer.load_state_dict(torch.load(checkpoint_path)['optimizer_state_dict'])
        final_epoch = (torch.load("stackgraphConvPool3DPnet/" + args.resume)['epoch'])+1
        train_optimisation(gpu, gpus, training_dataloader, test_dataloader, model, loss_fn, optimizer, None, dir_path, final_epoch)


def infer(gpu, gpus, world_size, checkpoint, file):
    torch.cuda.set_device(gpu)
    try:
        dist.init_process_group(backend='nccl')
    except RuntimeError:
        print("\n\nINFO:RuntimeError is raised >> Used gloo backend instead of nccl!\n")
        dist.init_process_group(backend='gloo')
    
    model = Classifier(shrinkingLayers, mlpClassifier)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    #load checkpoint
    checkpoint_path = "stackgraphConvPool3DPnet/" + checkpoint
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'], strict=False)
    
    test_data = ModelNet("ModelNet10_test_data", train=False, transform=lambda x: NormalizeScale()(NormalizeRotation()(SamplePoints(num=sample_points, remove_faces=True, include_normals=True)(x))))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=data_shuffle, num_workers=0, pin_memory=True, sampler=test_sampler)

    fpr = dict()
    tpr = dict()
    
    for batch_n, batch in enumerate(test_dataloader):
        X = torch.cat((batch.pos.cuda(non_blocking=True), batch.normal.cuda(non_blocking=True)), 1).view(batch_size, sample_points, -1) + torch.normal(
            torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points, dimensionality), fill_value=0.1)).cuda(gpu)
        
        pred = model(None, X)
        
        print(classes[items] for items in pred.argmax(1).tolist())
        if batch_n == 2:
            exit()

# -------------------------------------------MULTI-GPU MODEL---------------------------------------------------------- #

if __name__ == '__main__':
    gpus = torch.cuda.device_count()
    gpus = int(gpus)
    nodes = 1
    world_size = nodes * gpus
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--resume", type=str, required=False, help="Give relative path to certain checkpoint such as train56/epoch49.pth")
    parser.add_argument("--checkpoint", type=str, required=False, help="Give relative path to certain checkpoint such as train56/epoch49.pth")
    parser.add_argument("--data", type=str, required=False, help="Give absolute path to Point Cloud Data")
    parser.add_argument("--train_state", default=1, type=int, help="Set 1 for Train, 0 for Infer")
    args = parser.parse_args()
    
    #change state
    if args.train_state == True:
        train(args.local_rank, gpus, world_size)
    else:
        infer(args.local_rank, gpus, world_size, args.checkpoint, args.data)