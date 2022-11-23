All codes are in the testing phase yet!!!
It can be used after the test phase is completed with the permission of the author.

%% Written by Gizem Atac Kale and Cihan Karakuzu on 04.03.2019
%% Edited by Hüseyin PARMAKSIZ 21.11.2022 (for classification problems.)
clear all; close all; clc; warning off

% load fisheriris
% cs = categorical(species);
% ds = categories(cs);
% train_x = [];train_y = [];
% test_x = [];test_y = [];
% 
% for i = 1 : length(ds)
%     ind = find(cs == ds{i});
% % %     rand suffer
%     ind = ind(randperm(length(ind)));
% % %      75% training and 25% testing
%     train_x = [train_x; meas(ind(1:round(length(ind)*0.75)),:)];
%     train_y = grp2idx([train_y; grp2idx(cs(ind(1:round(length(ind)*0.75)),:))]);
%     test_x = [test_x; meas(ind(1+round(length(ind)*0.75):end),:)];
%     test_y = grp2idx([test_y; grp2idx(cs(ind(1+round(length(ind)*0.75):end),:))]);
% end
% X=train_x';
% Yd_train=train_y';
% Xt=test_x';
% Yd_test=test_y';

%------------------Loading datasets ----------------% 
%-----Load Ready-RF dataset -----------------------------%
%load('Ready9Ftrvete.mat');
%load('Ilk_RF_set.mat')
%-----Load Our RF dataset -------------------------------%
%load('Our_RF_tr-te_43F.mat');
%-----Load Our RF dataset after PCA ---------------------%
%load('Our_RF_43_PCA_tr-te_26F.mat');
%-----Load Our RF dataset after PCA and normalized-------%
%load('Our_RF_43_PCANormalize1-13_tr-te_19F.mat');
%-----Load Our RF dataset after MRMR select 8 features---%
%load('Our_RF_tr-te_43F_MRMR_8F.mat');
%load('Our_RF_tr-te_X_37F_70e30.mat');
%load('Our_RF_37_PCA_23F.mat')
load('Our_RF_tr-te_X_37F_70e30_MRMR_8F.mat');

Yd_train=train_data(:,1)'; X=train_data(:,2:size(train_data,2))';
Yd_test=test_data(:,1)';   Xt=test_data(:,2:size(test_data,2))';
%clear train_data;          clear test_data; %Release raw training & testing data array
%------------------Loading datasets ----------------% 

N=size(X,2);   Nt=size(Xt,2);%the number of samples for  test data
L=3; % number of layer number
Nnode=1000;%Nkume(jj);  % the number of neuron in each layer
DenSay=10; % Number of runs
Time=[];
Train_RMSE=[]; Test_RMSE=[]; % Cost array in terms of RMSE 
empty_solution.Ytrain=[];
empty_solution.Ytest=[];
solution=repmat(empty_solution,1);
ite=1;
Elm_Type = 1; REGRESSION=0; CLASSIFIER=1; 
%------------------Classification pre-processing ----------------% 
if Elm_Type~=REGRESSION
%%%%%%%%%%%% Preprocessing the data of classification
sorted_target=sort(cat(2,Yd_train,Yd_test),2);
label=zeros(1,1); % Find and save in 'label' class label from training and testing data sets
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:(N+Nt)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;
NofOutNeurons=number_class;
%%%%%%%%%% Processing the targets of training
temp_T=zeros(NofOutNeurons, N);
for i = 1:N
    for j = 1:number_class
        if label(1,j) == Yd_train(1,i)
            break;
        end
    end
temp_T(j,i)=1;
end
Yd_train=temp_T*2-1;
Yd_train=Yd_train';
%%%%%%%%%% Processing the targets of testing
temp_TV_T=zeros(NofOutNeurons, Nt);
for i = 1:Nt
    for j = 1:number_class
        if label(1,j) == Yd_test(1,i)
            break;
        end
    end
temp_TV_T(j,i)=1;
end
Yd_test=temp_TV_T*2-1;
Yd_test=Yd_test';
end
%------------------Classification pre-processing ----------------% 

while ite<=DenSay  %%%LOOP for numerical experiments 
tstart = tic;   % set clock time
W=[]; b=[]; Beta=[]; 
Xegt=X';  %for training set
Xtemp=Xegt;
Xtest=Xt'; %for testing set
%% SLFN structural parameters
n=size(X,1); %the number of external inputs
out_size=size(Yd_train,2);%the number of output
TrFcn = 'logsig';%transfer func of the node
C=1500;%Ckume(ii);  %regularisation factor
%% Stage:1 Autoencoders learning or representation learning
for m=1:L-1 %do it for each layer
%% construction of mth SLFN and initial assigmants 
    W=randn(Nnode,size(Xtemp,2))'; %input weights of hidden nodes
    W=orth( W.' ).';% orthogonal yap�yor
    iw{m}=W;% collect the input weights
    b=randn(Nnode,1)';
    b=orth(b.').';  %orthogonal biases of neurons
    ib{m}=b;  % collect neuron biases 
%% construction of H matrice
    H=feval(TrFcn,Xtemp*W+repmat(b,N,1));  
    Beta=H'*inv(eye(size(H*H'))/C+H*H')*Xtemp;
    iBeta{m}=Beta;
    Xtemp=(Xtemp*Beta');  
end % for m
%% Stage:2 Training of the last layer of ELM
% H computation for the last layer
    Wfinal=randn(Nnode,size(Xtemp,2))'; %input weights of hidden nodes
    Wfinal=orth(Wfinal.').';% ortonormal assigned
    bfinal=randn(Nnode,1)';  %biases of neurons
    bfinal=orth(bfinal.').';
    Hfinal=feval(TrFcn,Xtemp*Wfinal+repmat(bfinal,N,1)); 
    Beta_final=Hfinal'*inv(eye(size(Hfinal*Hfinal'))/C+Hfinal*Hfinal')*Yd_train;
    Time(ite)= toc(tstart);
% PERFORMANCE FOR TRAINING DATA with founded output weight (Beta)parameters 
    Ytrain= Hfinal*Beta_final;
    Y=Ytrain';
    %---solution.Ytrain(ite,:)=Ytrain;
% Cost computatiton for training 
    Cost_train=sqrt(mse(Ytrain,Yd_train)); 
% PERFORMANCE FOR TESTING DATA with founded parameter 
    Xmt=Xtest;
    for m=1:L-1 %do it for each layer
        Xmt=Xmt*(iBeta{1,m})';  
    end % for m
    HTfinal=feval(TrFcn,(Xmt*Wfinal+repmat(bfinal,Nt,1))); 
    Ytest= HTfinal*Beta_final;
    TY=Ytest';
    %--solution.Ytest(ite,:)=Ytest;
% Cost computatiton for training 
    Cost_test=sqrt(mse(Ytest,Yd_test));
% Cost saving in an array
     Train_RMSE(ite)=Cost_train; Test_RMSE(ite)=Cost_test;
ite=ite+1; %iteration increment
end % while ite 

%% %%%%%%%%%% Calculate training & testing classification accuracy
if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
RateofMissClass=0; RateofMissClassT=0;
Yd_train=Yd_train'; Yd_test=Yd_test';
for i = 1 : size(Yd_train, 2)
    [x, i_Labeld]=max(Yd_train(:,i));
    [x, i_labela]=max(Y(:,i));
    if i_labela~=i_Labeld
        RateofMissClass=RateofMissClass+1;
    end
end
TrainingAccuracy=1-RateofMissClass/size(Yd_train,2);
for i = 1 : size(Yd_test, 2)
    [x, i_Labeld]=max(Yd_test(:,i));
    [x, i_labela]=max(TY(:,i)); % TY: the actual output of the testing data
    if i_labela~=i_Labeld
        RateofMissClassT=RateofMissClassT+1;
    end
end
TestingAccuracy=1-RateofMissClassT/size(Yd_test,2);
end
%%
% displaying the training time metrics 
disp('[-Elapsed time metrics for training-]'); 

average_time=mean(Time);
best_time=min(Time);
worst_time=max(Time);
std_dev_time=std(Time);

disp('[Mean      Best      Worst     StdDev]'); 
Mtrc=[average_time best_time worst_time std_dev_time];
disp(Mtrc);

disp('[-Accuracy for training-]'); 
MtrcTr=[TrainingAccuracy];
disp(MtrcTr);
disp('[-Accuracy for testing-]'); 
MtrcTe=[TestingAccuracy];
disp(MtrcTe);

% disp('The metrics for training data'); 
% average_cost_train=mean(Train_RMSE);
% [best_cost_train,best_run_train]=min(Train_RMSE);
% [worst_cost_train, worst_run_train]=max(Train_RMSE);
% std_dev_cost_train=std(Train_RMSE);
% disp('[  MeanRMSE  BestRMSE  WorstRMSE   StdDev]'); 
% Mtrc_train=[average_cost_train best_cost_train worst_cost_train std_dev_cost_train];
% disp(Mtrc_train);
% 
% disp('The metrics for testing data'); 
% average_cost_test=mean(Test_RMSE);
% [best_cost_test,best_run_test]=min(Test_RMSE);
% [worst_cost_test, worst_run_test]=max(Test_RMSE);
% std_dev_cost_test=std(Test_RMSE);
% 
% disp('[  MeanRMSE  BestRMSE  WorstRMSE   StdDev]'); 
% Mtrc_test=[average_cost_test best_cost_test worst_cost_test std_dev_cost_test];
% disp(Mtrc_test); 

save results\ML-ELM-L3-Classification.mat
