function [C] = clustering (inputs, Ctype, Cnum)
%---------------------------------
% Cnum : the number of expected clusterings
% Ctype=1 : K-means
% Ctype=2 : Gaussian Mixture Model （GMM�?
% Ctype=3�? Fuzzy C means (FCM)
% Ctype=4 : Possibilistic C-Means （PCM�?
% Ctype=5 : Possibilistic Fuzzy C-Means（PFCM�?
% Ctype=6 : Enhanced Possibilistic Fuzzy C-Means（EPFCM�?
% Ctype=7 : Interval Type 2 Possibilistic Fuzzy C-Means（IT2-EPFCM�?
%---------------------------------

% %% Initialization for EPFCM and IT2-EPFCM (Optional)
% [V,U] = fcm(Xin,nC,[NaN 100 0.0001 0]);
% ETA_init = Initialization_ETA (Xin, U, V, mean(m1,m2), K);

%% Clustering
if Ctype == 1 % (k-means)    
  [C.idx,C.center]=kmeans(inputs,Cnum);
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end

if Ctype == 2 % (GMM) 
  gmModel = fitgmdist(inputs, Cnum);
  C.index = cluster(gmModel, inputs);
  C.center = gmModel.mu;
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end

if Ctype == 3 % (FCM)  
  options = [2.0, 1000, 1e-5, 0];
  [C.center,U,~] = fcm(inputs,Cnum,options);
  [rows, ~] = size(inputs);
  C.idx=zeros(rows,1);

  if Cnum == 1
    % 处理仅一个簇的情况，例如将所有数据点标记为一个簇
    C.idx = ones(size(U, 2), 1);
  else
      for ii=1:rows
          C.idx(ii) = find(U(:,ii)==max(U(:,ii)));
      end
  end
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end
if Ctype == 4 % (PCM)  *********未完成，报错！！�?***********
  options = [2, 100, 1e-5, 1];
  [C.center,U,~] = pcm(inputs,Cnum, options);
  [rows, ~] = size(inputs);
  C.idx=zeros(rows,1);
  for ii=1:rows
      C.idx(ii) = find(U(:,ii)==max(U(:,ii)));
  end
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end

if Ctype == 5 % (PFCM) *********未完成，报错！！�?***********
  options = [2.0, 2.0, 100, 1e-5, 1];
  [C.center, U, ~] = pfcm(inputs,Cnum, options);
  [rows, ~] = size(inputs);
  C.idx=zeros(rows,1);
  for ii=1:rows
      C.idx(ii) = find(U(:,ii)==max(U(:,ii)));
  end
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end

%*****************************************
%*************Not finish******************
%*****************************************

if Ctype == 6 % (EPFCM) 

% Options --------------------------------------------------------------
  m = 2;
  Theta = 3;
  Cf=0.5;
  Cp=0.5;
  % Initialization for EPFCM (Optional)
  [V,U] = fcm(inputs,Cnum,[NaN 200 0.0001 0]);
  ETA_init = Initialization_ETA (inputs, U, V, m);
  [C.center,U_EPFCM,T_EPFCM,E,ObjFun_EPFCM] =EPFCM_clustering (inputs,Cnum,m,Theta,Cf,Cp,ETA_init);
  [rows, ~] = size(inputs);
  C.idx=zeros(rows,1);
  for ii=1:rows
    C.idx(ii) = find(U_EPFCM(:,ii)==max(U_EPFCM(:,ii)));
  end
  [C.std] = Cstd(inputs, Cnum, C.center,C.idx);
end

if Ctype == 7 % (IT2-EPFCM) 
  m1 = 2.0;
  m2 = 4.0;
  Theta1 = 3.0;
  Theta2 = 5.0;
  Cf=0.5;
  Cp=0.5;
% Initialization for IT2-EPFCM (Optional)
[V,U] = fcm(inputs,Cnum,[NaN 200 0.0001 0]);
ETA_init = Initialization_ETA (inputs, U, V, mean(m1,m2));
% IT2-EPFCM ------------------------------------------------------------
[V1,V2,U1,U2, E1,E2] = IT2_EPFCM_clustering (inputs,Cnum,m1,m2,Theta1,Theta2,Cf,Cp,ETA_init);
x=1;
end

%% Computing the std／width
function [std] = Cstd(data, num, center,idx)
    for i = 1:num
        points_in_cluster = data(idx == i, :); 
        var=(sum((points_in_cluster - center(i, :)).^2))/size(points_in_cluster, 1);
        std(i,:) = sqrt(var);            
    end
end

end
