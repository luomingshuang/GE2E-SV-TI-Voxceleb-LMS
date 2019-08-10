#coding:utf-8
import os
import random
import time
import torch
from torch.utils.data import DataLoader

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, GE2ELoss, get_centroids, get_cossim
from tensorboardX import SummaryWriter

model_path = './speech_id_checkpoint/ckpt_epoch_360_batch_id_281.pth'

if (__name__=='__main__'):

    writer = SummaryWriter()

    device = torch.device(hp.device)
    model_path = hp.model.model_path

    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed(hp.data.train_path, hp.train.M)
    else:
        train_dataset = SpeakerDatasetTIMIT(hp.data.train_path, hp.train.M)
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed(hp.data.test_path, hp.test.M)
    else:
        test_dataset = SpeakerDatasetTIMIT(hp.data.test_path, hp.test.M)

    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)

    embedder_net = SpeechEmbedder().to(device)

    
    embedder_net.load_state_dict(torch.load(model_path))
    ge2e_loss = GE2ELoss(device)
    #Both net and loss have trainable parameters
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    
    embedder_net.train()

    with torch.no_grad():
        avg_EER = 0
        iteration = 0
        for e in range(hp.test.epochs):
            batch_avg_EER = 0
            for batch_id, mel_db_batch in enumerate(test_loader):
                assert hp.test.M % 2 == 0
                enrollment_batch, verification_batch = torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
            
                enrollment_batch = torch.reshape(enrollment_batch, (hp.test.N*hp.test.M//2, enrollment_batch.size(2), enrollment_batch.size(3))).cuda()
                verification_batch = torch.reshape(verification_batch, (hp.test.N*hp.test.M//2, verification_batch.size(2), verification_batch.size(3))).cuda()
            
                perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
                unperm = list(perm)
                for i,j in enumerate(perm):
                    unperm[j] = i
                
                verification_batch = verification_batch[perm]
                enrollment_embeddings = embedder_net(enrollment_batch)
                verification_embeddings = embedder_net(verification_batch)
                verification_embeddings = verification_embeddings[unperm]
            
                enrollment_embeddings = torch.reshape(enrollment_embeddings, (hp.test.N, hp.test.M//2, enrollment_embeddings.size(1)))
                verification_embeddings = torch.reshape(verification_embeddings, (hp.test.N, hp.test.M//2, verification_embeddings.size(1)))
            
                enrollment_centroids = get_centroids(enrollment_embeddings)
            
                sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
                # calculating EER
                diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
                for thres in [0.01*i+0.5 for i in range(50)]:
                    sim_matrix_thresh = sim_matrix>thres
                
                    FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                    /(hp.test.N-1.0)/(float(hp.test.M/2))/hp.test.N)
    
                    FRR = (sum([hp.test.M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(hp.test.N))])
                    /(float(hp.test.M/2))/hp.test.N)
                
                    # Save threshold when FAR = FRR (=EER)
                    if diff> abs(FAR-FRR):
                        diff = abs(FAR-FRR)
                        EER = (FAR+FRR)/2
                        EER_thresh = thres
                        EER_FAR = FAR
                        EER_FRR = FRR
                batch_avg_EER += EER
                print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR))
            avg_EER += batch_avg_EER/(batch_id+1)
            writer.add_scalar('data/EER', batch_avg_EER/(batch_id+1), iteration)
            iteration += 1
        avg_EER = avg_EER / hp.test.epochs
        print("\n EER across {0} epochs: {1:.4f}".format(hp.test.epochs, avg_EER))
    writer.close()
