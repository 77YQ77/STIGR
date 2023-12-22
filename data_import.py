import numpy as np
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
file_path_ASD = 'Data/ASD_preprocessed/ASD_cc200'
file_path_ADHD = 'Data/ADHD_preprocessed/ADHD_cc200'


def import_data(dataset):
    print('start collecting ' + dataset + ' data! ')
    Num_ASD_Phenotypic = 1112  # don't change
    Num_ADHD_Phenotypic = 973  # don't change

    a = 1
    b = -1
    # import data
    data = []
    if dataset == 'ASD':
        for file in os.listdir(file_path_ASD):
            print(file)
            x = np.array(np.loadtxt(os.path.join(file_path_ASD,file)))
            for i in range(x.shape[0]):
                r_mean = np.sum(np.abs(x[i,:])) / x.shape[1]
                for j in range(x.shape[1]):
                    if r_mean == 0:
                        x[i, j] = 0
                    else:
                        x[i, j] = x[i, j] / r_mean
            data.append(x)

    if dataset == 'ADHD':
        for file in os.listdir(file_path_ADHD):
            x = np.array(np.loadtxt(os.path.join(file_path_ADHD,file)))
            for i in range(x.shape[0]):
                r_mean = np.sum(np.abs(x[i, :])) / x.shape[1]
                for j in range(x.shape[1]):
                    if r_mean == 0:
                        x[i, j] = 0
                    else:
                        x[i, j] = x[i, j] / r_mean
            data.append(x)

    # labels
    if dataset == 'ASD':
        os.chdir('Data/ASD_preprocessed/ASD_cc200')
        df = pd.DataFrame(pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv'))

    if dataset == 'ADHD':
        os.chdir('Data/ADHD_preprocssedADHD_cc200')
        df = pd.DataFrame(pd.read_csv('../adhd200_preprocessed_phenotypics.tsv',sep='\t'))


    file_list = []
    for root, dirs, files in os.walk("."):
        file_list = files

    print(file_list)

    os.chdir('../../../')
    labels = np.zeros(len(file_list))
    Sex = []
    Age = []
    Handedness = []
    IQ = []

    if dataset == 'ASD':
        A = np.zeros((Num_ASD_Phenotypic, 6))
        for i in range(Num_ASD_Phenotypic):
            A[i][0] = df.values[i, 0]  # subject ID
            A[i][1] = df.values[i, 5]  # whether autism
            A[i][2] = df.values[i, 7]  # subject age
            A[i][3] = df.values[i, 8]  # subject sex
            A[i][4] = df.values[i, 9]  # handedness
            A[i][5] = df.values[i, 11]  # IQ

        for i in range(len(file_list)):
            for j in range(Num_ASD_Phenotypic):
                if str(int(A[j][0])) in file_list[i]:
                    if A[j, 1] == 1:  # ASD
                        labels[i] = 1
                    else:  # TC
                        labels[i] = 0
                    Age.append(A[j,2])
                    if A[j,3] == 2:
                        Sex.append(0)
                    else:
                        Sex.append(A[j,3])
                    if A[j,4] == None:
                        Handedness.append(1)
                    else:
                        Handedness.append(A[j,4])
                    if A[j,5] == None:
                        IQ.append(100)
                    elif A[j,5] == -9999:
                        IQ.append(100)
                    else:
                        IQ.append(A[j,5])
                    break
    if dataset == 'ADHD':
        A = np.zeros((Num_ADHD_Phenotypic, 6))
        for i in range(Num_ADHD_Phenotypic):
            A[i][0] = df.values[i, 0]  # subject ID
            A[i][1] = df.values[i, 5]  # DX 0: TC 1: ADHD-combined 2: ADHD-Hyperactive/Impulsive 3: ADHD-Inattentive, pending = -1
            A[i][2] = df.values[i, 2]  # Sex
            A[i][3] = df.values[i, 3]  # Age
            A[i][4] = df.values[i, 4]  # Handedness
            A[i][5] = df.values[i, 15]  # IQ

        for i in range(len(file_list)):
            for j in range(Num_ADHD_Phenotypic):
                if str(int(A[j][0])) in file_list[i]:

                    if A[j, 1] == 0:  # TC
                        labels[i] = 0
                    else:  # ADHD
                        labels[i] = 1
                    if A[j, 2] == None:
                        Sex.append(1)
                    else:
                        Sex.append(A[j, 2])
                    Age.append(A[j, 3])
                    if A[j, 4] == None:  # 1 right hand, -1 left hand, 0 hybrid
                        Handedness.append(1)
                    elif np.round(A[j, 4]) == 0 or A[j, 4] < 0:
                        Handedness.append(-1)
                    elif np.round(A[j, 4]) == 1:
                        Handedness.append(1)
                    else:
                        Handedness.append(0)
                    if A[j, 5] == -999:
                        IQ.append(100)
                    elif A[j, 5] == None:
                        IQ.append(100)
                    else:
                        IQ.append(A[j, 5])
                    break

    return np.array(data),labels,Sex,Age,Handedness,IQ

if __name__ =='__main__':
    # import fMRI data
    data, labels, Sex, Age, Handedness, IQ = import_data('ADHD')
    np.save('Data/ADHD/ADHD_feats_norm.npy',data)
    np.save('Data/ADHD/ADHD_labels_norm.npy',labels)
    
    data = data[:,:-1,:]

    a = -1
    b = 1
    Age = a + (b - a) / (max(Age) - min(Age)) * (Age - min(Age))
    IQ = a + (b - a) / (max(IQ) - min(IQ)) * (IQ - min(IQ))
    
    ADHD_phenotype = np.mat(np.vstack((np.array(Age), np.array(IQ), np.array(Sex), np.array(Handedness)))).T
    np.save('Data/ADHD/ADHD_phenos_norm.npy',ADHD_phenotype)


