% cluster analysis
function [cl] = ca(type, trndat, cn)

if type == 1 % HCM (k-means)
    
  [cl.idx,cl.center]=kmeans(trndat,cn);
  U = cafuzz(trndat, cn, cl.center);
  [clmean,clvariance,clstd,clmin,clmax] = castat(trndat, cn, cl.idx);

end

if type == 2 % FCM (fuzzy c-means)
    
  [cl.center,U,OF] = fcm(trndat,cn);
  [rows, cols] = size(trndat);
  cl.idx=zeros(rows,1);
  for i=1:rows
      cl.idx(i) = find(U(:,i)==max(U(:,i)));%所有聚类中隶属度最大的那个
  end
  [clmean,clvariance,clstd,clmin,clmax] = castat(trndat, cn, cl.idx)
  
end

if type == 3 % HCA (hclust)
    
    Y=pdist(trndat);
    Z=linkage(Y);
    T = cluster(Z,'maxclust',cn);
    center = [];
    for i=1:cn

     %idx: finding index and data of the next cluster samples
     idx=find(T(:,1)==i);
     cldat=trndat(idx,:);
     [r,c] = size(cldat);
     if r==1
      % when cluster contains only 1 sample
      center=[center; mean(cldat,1)];
     else
      center=[center; mean(cldat)];
     end

    end
  
    cl.idx = T(:,1);
    cl.center = center;
    
    U = cafuzz(trndat, cn, cl.center);
    [clmean,clvariance,clstd,clmin,clmax] = castat(trndat, cn, cl.idx);

end

cl.U = U;
cl.mean = clmean;
cl.variance = clvariance;
cl.std = clstd;
cl.min = clmin;
cl.max = clmax;

% MEAN ====================================================================
Mcenter = [];
[row,col] = size(trndat);
numInp = col
cnum = cn
% -------------------------------------------------------------------------
for i=1:cnum
Mcenter = [Mcenter cl.center(i,1:numInp)];
end
% -------------------------------------------------------------------------
M = reshape(Mcenter,numInp,cnum)';

% makes fuzzification based on distances between center and samples
function [U] = cafuzz(data, cluster_n, center)
    expo = 2;
    dist = distfcm(center, data); % fill the distance matrix
    tmp = dist.^(-2/(expo-1));  % calculate new U, suppose expo != 1
    U = tmp./(ones(cluster_n, 1)*sum(tmp));
    U(isnan(U))=0;
    
function [clmean,clvariance,clstd,clmin,clmax] = castat(trndat,cn,clidx)

    [r,c] = size(trndat(1,:));
        
    for i=1:cn

       idx=find(clidx==i);
       cldat=trndat(idx,:);
       [rc,cc] = size(cldat); 

       if ~isempty(cldat)
           if rc > 1
               clmean(i,1:c) = mean(cldat);
               clvariance(i,1:c) = var(cldat);
               clstd(i,1:c) = sqrt(clvariance(i,:));
               clmin(i,1:c) = min(cldat);
               clmax(i,1:c) = max(cldat);
           else
               clmean(i,1:c) = cldat;
               clvariance(i,1:c) = (1e-12)^2;
               clstd(i,1:c) = 1e-12;
               clmin(i,1:c) = cldat;
               clmax(i,1:c) = cldat;
           end
       else
           clmean(i,1:c) = 0;
           clvariance(i,1:c) = 0;
           clstd(i,1:c) = 0;
           clmin(i,1:c) = 0;
           clmax(i,1:c) = 0;
       end % if ~isempty(cldat)

    end % for i=1:cn
    

