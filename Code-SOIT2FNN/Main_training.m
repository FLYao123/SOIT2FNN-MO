%% SOIT2FNN-MO training stage
% Please read the README file first before running the code.

%% Clear MATLAB cache
clc;
clear all;
close all;

%% Load dataset (Already normalized)
subfolder_data = 'IT2FNN_normalization_data';
%1) Load chaotic time series
filename_train = fullfile(subfolder_data, 'Chaotic_train_9_3_0.000000.mat');
load(filename_train)
X = train_input';
Y = train_output';

% %%2) Load microgrid time series
% filename_train = fullfile(subfolder_data, 'unmets_HWM_train.mat'); %Alternatively, use prs_HWM_train.mat.
% load(filename_train)
% inputs = trainData(:, 1:15);
% X = inputs(:,1:12);
% Y = inputs(:,13:15);

%% Clustering (If there is no pool of initial fuzzy rules)
%---------------------------------
% Cnum : the number of expected clusterings
% Ctype=1 : K-means
% Ctype=2 : Gaussian Mixture Model 
% Ctype=3 : Fuzzy C means (FCM) 
% Ctype=4 : Possibilistic C-Means 
% Ctype=5 : Possibilistic Fuzzy C-Means
% Ctype=6 : Enhanced Possibilistic Fuzzy C-Means
% Ctype=7 : Interval Type 2 Possibilistic Fuzzy C-Means
%---------------------------------
Ctype = 3;
Cnum = 5; % 10, 15, 20, 25, 30, ...

% % ******* Clustering  & Save for SOIT2FNN-MO******
% %For Layer 2
% [C_x] = clustering (X, Ctype, Cnum);
% % save('unmets_12_3_clustering_FCM_5.mat', 'C_x'); 
% save('Chaotic_9_3_clustering_FCM_5_0.0.mat', 'C_x');
% 
% %For Layer 4
% [C_mid] = clustering (X, Ctype, 1);
% % save('unmets_12_3_clustering_FCM_5_mid.mat', 'C_mid');
% save('Chaotic_9_3_clustering_FCM_mid_0.0.mat', 'C_mid');

% Note: Please remember to move the saved clustering files to the subfolder **subfolder_data** (i.e., IT2FNN_normalization_data) directory.
%% Clustering  & Loading (If there is a pool of initial fuzzy rules)
% %For microgrid 
% filename_cluster = fullfile(subfolder_data, 'unmets_9_3_clustering_FCM_5.mat');
% filename_cluster_mid = fullfile(subfolder_data, 'unmets_9_3_clustering_FCM_10_mid.mat');
%For Chaotic
filename_cluster = fullfile(subfolder_data, 'Chaotic_9_3_clustering_FCM_5_0.0.mat');
filename_cluster_mid = fullfile(subfolder_data, 'Chaotic_9_3_clustering_FCM_mid_0.0.mat');
load(filename_cluster);
load(filename_cluster_mid);

%% Parameter initialization

I = size(X,2); % i.e., the number of inputs (n) in the paper
M = size(Y,2); % i.e., the number of outputs (K) in the paper

%For antecedent parameters, as shown in equations (16) and (17) of the paper.
uncertainty = 0.1; % uncertainty in the mean value, alongside equation (16) in the paper
c_up = C_x.center * (1+uncertainty); 
c_lo = C_x.center * (1-uncertainty);
c_mid = C_mid.center;
sigma = C_x.std; 
sigma_mid = C_mid.std; 
%Other parameter nitialization
buffer_size = size(c_lo,1);
Rule_num = 0;% i.e., M in the paper
ant_par = [];%matrix to contain antecedent parameters
con_par = [];%matrix to contain consequent parameters
Mqlqr = zeros(M, 2) + 0.5; % i.e., q_l^k and q_r^k in (12) and (13)
Mylyr = zeros(1,M) + 0.5; % i.e.,  q_o^k in (14)
 % Mant = [c_mid*ones(M,1), sigma_mid*ones(M,1)];
M_c_mid = repmat(c_mid, M, 1); 
M_sigma_mid = repmat(sigma_mid, M, 1);
Mant = cat(3, M_c_mid, M_sigma_mid); % matrix to contain co-antecedent parameters
res_last = 1e10; %1000
threshold_growing = 0.0025; %alternatively->0.001
threshold_reducing = threshold_growing*1; %threshold for growing and removing a rule
K_link = 0.1; % initinal parameter in Layer 9
Flag = 0; % In this code, to track the execution process for storing intermediate parameters (such as potential consequent parameters), and restoring the rule pool, the values of the Flag differ from those described in the manuscript. Don't worry, the overall logic and structure is the same. Please refer to the manuscript for a better understanding of the code. We will address this difference before submitting the code to GitHub.
lr = 0.03; % learning rate for stage 2
ep_max = 3000; % maximum episode for stage 2 , alternatively->5000
batch_size = 64; % batch size for stage 2
replay_momory = [];

for epo = 1:1000%use a large value that ensures multiple iterations through the pool of initial fuzzy rules.
    con_par_stage1 = [];
    Mqlqr_growing = [];
    Mylyr_growing = [];
    Mant_growing = [];
    K_link_growing = [];
    res = zeros(1,size(c_up,1));
    if isempty(c_up)
       break;
    end
% ****** First stage: Rule growing ******
    for i = 1:size(c_up,1)   
        seed = 42;
        rng(seed);
        ant_tmp_tuple= cat(3, c_lo(i,:), c_up(i,:), sigma(i,:));
        c = randn(I + 1, M);
        s = randn(I + 1, M);
        con_tmp_tuple = cat(3, c,s);
   
        if isempty(ant_par) ==1%If this is the first rule
            ant_par_tmp = ant_tmp_tuple;
            con_par_tmp = con_tmp_tuple;
        else
            ant_par_tmp = cat(4, ant_par,ant_tmp_tuple);
            con_par_tmp = cat(4, con_par,con_tmp_tuple);
       end

        % Computing the consequent parameters
        [res_tmp, con_par_tmp, Mqlqr_tmp,Mylyr_tmp,Mant_tmp,K_link_tmp] = Rule_growing(X,Y, ant_par_tmp,Mant, con_par_tmp,Mqlqr,I,M, K_link,Mylyr); %ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â¸ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¤ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â±ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¡ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â½ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦�?????Ãƒâ€šÃ�?�Â¢ÃƒÆ�?�Ã¢â�?��??? ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢Ã¢â�?��????? ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¨ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Â¹Ãƒ??Ã‚Â ÃƒÆ�?�Ã�?�Â¢ÃƒÂ�??Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â©ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢Ã¢â�?�¬Å¡Ã�?�Â¬Ãƒâ€šÃ‚Â¡ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¦ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢Ã¢�??šÂ¬Ã…Â¡Ãƒ�????? ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¯ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ�?��???Â¹ÃƒÂ¢Ã¢â€šÂ¬Ã�?�Â ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¿ÃƒÆ�?�Ã�?�Â¢�????Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¤ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂªÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â¡�?????Ãƒâ€¦Ã�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?��???ÃƒÂ¢Ã¢â€šÂ¬Ã�?�ÂÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¶ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â�?????Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â®ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?��???Â¹Ãƒâ€¦�???Å“ÃƒÆ�?�Ã�??Å¡Ãƒâ€šÃ�?�Â¯ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?��???Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¤�??????ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂªÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥�?????Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã�?��???Ãƒâ€šÃ�?�Â score_tmpÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¯ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ???
        res(i) = mean(res_tmp); 
        con_par_stage1 = cat(5,con_par_stage1, con_par_tmp); 
        Mqlqr_growing = cat(3,Mqlqr_growing,Mqlqr_tmp);  
        Mant_growing  = cat(4,Mant_growing,Mant_tmp);
        K_link_growing = cat(2,K_link_growing,K_link_tmp);
    end
    
      [minres, index] = min(res); 
      % rule_growing_res(epo) = res(index);
      % [maxscore, index] = max(score); 
      
    if (res_last - minres) >= threshold_growing  % Rule growing
        ant_tuple= cat(3, c_lo(index,:), c_up(index,:), sigma(index,:));
        ant_par = cat(4, ant_par,ant_tuple);
        con_par = con_par_stage1(:,:,:,:,index);
        Mqlqr = Mqlqr_growing(:,:,index);
%         Mylyr = Mylyr_growing(:,:,index);
        Mant = Mant_growing(:,:,:,index);
        K_link = K_link_growing(:,index);
        Rule_num = Rule_num +1;
        Flag = 1;
        res_last = minres;
    else
        if size(ant_par,4) == 0 || size(ant_par,4) == 1
            if Flag == 3 
                Flag = 4;
            else
                Flag =3; 
            end
        else
            [res_reducing_tmp, rule_reducing_index, Mqlqr_red, con_par_red,Mylyr_red,Mant_red,K_link_red] = Rule_Reducing(X,Y, ant_par,con_par,Mqlqr,I,M,res_last,Mant, K_link,Mylyr); % ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â³ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¨ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¯ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¼ÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¯ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¬ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂªÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¡ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â°�?????Ãƒâ€¹Ã�?��??œÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢Ã¢�??šÂ¬Ã…Â¡Ãƒ�??šÃ‚Â¬ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂªÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â§ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��??????,ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂªÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¿ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥�?????Ãƒâ€šÃ�?�ÂºÃƒÆ�?�Ã¢â�?�¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¸ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢Ã¢�??šÂ¬Ã…Â¡Ãƒ�??šÃ‚Â¬ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Âªres_tmp
         
            if res_reducing_tmp - res_last < threshold_reducing 
                if Flag == 3 
                     Flag = 4;
                else
                     Flag =3; 
                end
            else
               c_up(end+1,:) =  replay_momory(:,:,1,rule_reducing_index);
               c_lo(end+1,:) = replay_momory(:,:,2,rule_reducing_index);
               sigma(end+1,:) = replay_momory(:,:,3,rule_reducing_index);
               ant_par(:, :, :, rule_reducing_index) = [];
               % con_par(:, :, :, rule_reducing_index) = [];
               con_par = con_par_red;               
               Mqlqr = Mqlqr_red;
%                Mylyr = Mylyr_red;
               Mant = Mant_red;
               K_link = K_link_red;

               res_last = res_reducing_tmp;
               Rule_num = Rule_num - 1;
               % stage2_flag = 0;
               Flag = 2;
            end        
        end
    end
    % if flag_rule_reducing == 1%ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â­ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¤ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ã�???Ã‚ÂÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒ�??šÃ‚Â¶ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â§�?????Ãƒâ€¦Ã�?�Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã¢â�?�¬Â¹ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ�?�Ã�?�Â¢Ãƒ�???Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¢ÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â¾ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¯ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¼ÃƒÆ�?��???Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¹ÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â¸ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¸ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥�?????Ãƒâ€šÃ�?�Â ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��????Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â°ÃƒÆ�?��????Ãƒâ€¹Ã�?��??œÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¯ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¼ÃƒÆ�?��???Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Â¹Ãƒ??Ã‚Â ÃƒÆ�?�Ã�?�Â¢ÃƒÂ�??Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã¢â�?�¬Å¡Ã�????ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¥ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â§ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¹ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â°ÃƒÆ�?��???Â Ãƒ??Ã¢â€žÂ¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦�?????Ãƒâ€šÃ�?�Â¢ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â´ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?��????Ãƒâ€¦Ã�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?��???Ãƒâ€šÃ�?�Â¢ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¯ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¼ÃƒÆ�?��???Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¿ÃƒÆ�?�Ã�?�Â¢�????Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â¡ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â°ÃƒÆ�?��???Â Ãƒ??Ã¢â€žÂ¢ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¢ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â´ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��????Ãƒâ€šÃ�?�Â°ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?��???Â¦Ãƒâ€šÃ�?�Â½ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¤ÃƒÆ�?��??????ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¨ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚ÂµÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â·ÃƒÆ�?�Ã�?��??™Ãƒ�??šÃ‚Â¨ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�Â°ÃƒÆ�?��???Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¦ÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã�???? 
    if Flag == 3 || Rule_num == Cnum
        [ant_par_t,Mant_t,con_par_t,Mqlqr_t,Mylyr_t,K_link_t, loss] = stage2_all_pars_tuning(X, Y, ant_par,Mant,con_par,Mqlqr,I,M,lr, ep_max,batch_size, K_link,Mylyr);   % Stage 2 ÃƒÆ’Ã�?�â€™Ãƒ�??šÃ‚Â¨ÃƒÆ�?�Ã¢â�?�¬Å¡Ãƒâ€šÃ‚Â°ÃƒÆ�?��???Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢�??žÂ¢ÃƒÆ’Ã�?��??™Ãƒ�??šÃ‚Â¥ÃƒÆ�?��???Å¡Ãƒâ€šÃ�?�ÂÃƒÆ�?�Ã�?�Â¢ÃƒÂ¢�???Å¡Ã�???
        if loss(end) > res_last
            msgbox('Stage2 does not work! Please have a check! Keep the previous record!', 'Warning!');
        else
	        if  res_last - loss(end)<threshold_growing
               msgbox('Stage 2 does not work well! A little help! Maybe not better than add a new rule','Warning!');
            end
            ant_par = ant_par_t;
            Mant = Mant_t;
            con_par = con_par_t;
            Mqlqr = Mqlqr_t;
%             Mylyr = Mylyr_t;
            K_link = K_link_t;

            res_last = loss(end);
        end
    end

    if Flag == 1 % 
        replay_momory_tmp = cat(3, c_up(index,:),  c_lo(index,:),sigma(index,:));
        c_up(index,:) = []; 
        c_lo(index,:) = []; 
        sigma(index,:) = []; 
        replay_momory = cat(4,replay_momory, replay_momory_tmp); 
    elseif Flag == 4 
        break;
    end

rule_number(epo) = Rule_num;

end

% disp(rule_growing_res)
% total_time = toc;
% disp(['total training time costs: ', num2str(total_time)]);

% disp(loss(end))
figure(100)
plot(rule_number)

if Rule_num ~= Cnum
    figure(1)
    plot(loss)
else
    disp('Cnum = Rule_num');
    disp(loss(end))
end

%% Save network weights
subfolder_weights = 'Weights';

filename_weights = fullfile(subfolder_weights, sprintf('IT2FNN3_Chaotic_0.0_weights_I%d_M%d_C%d_Tg%2f_Tr%2f.mat', I, M,Cnum,threshold_growing,threshold_reducing));
% filename_weights = fullfile(subfolder_weights, sprintf('IT2FNN3_unmets_weights_I%d_M%d_C%d_Tg%2f_Tr%2f.mat', I-3, M,Cnum,threshold_growing,threshold_reducing));
save(filename_weights, 'ant_par', 'con_par','Mqlqr','Mant','Mylyr', 'K_link','rule_number');


