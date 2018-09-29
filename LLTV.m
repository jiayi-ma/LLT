function VecFld = LLTV(X, Y, conf)
% LLTV  Locally Linear Trasforming for nonrigid
%   VECFLD = LLTV(X, Y, CONF)
%   learning vector field from random samples with outliers.
%   Input:
%   X, Y: Original data.
%   conf: Refer to the function LLT_init
%% Initialization
gamma = conf.gamma;
beta = conf.beta;
eta = conf.lambda;
theta = conf.theta;
a = conf.a;
MaxIter = conf.MaxIter;
ecr = conf.ecr;
minP = conf.minP;
M = conf.M;
Kn = conf.Kn;

%% Initialization gamma, C = 0,P = I_N*N
[N, D]=size(X); 
tmp_X = unique(X, 'rows'); idx = randperm(size(tmp_X,1)); 
idx = idx(1:min(M,size(tmp_X,1)));ctrl_pts=tmp_X(idx,:);
K=con_K(ctrl_pts,ctrl_pts,beta);% M*M
U = con_K(X, ctrl_pts, beta);%N*M
C = zeros(M, D);%M*2
P = eye(N);%N*N
iter=1;  tecr=1; E=1;
sigma2=sum(sum((Y-X).^2))/(N*D);

%%
% Search the k nearest neighbors for each point X
X2= sum(X.^2,2);   %N*1
distance = repmat(X2,1,N)+repmat(X2',N,1)-2*X*X';
[~, index] = sort(distance);
neighborhood = index(2:(1+Kn),:); %K*N

%%
% Compute W by minimizeing the cost function E_LLE(W) = sum(square(xi - sum(Wij*xj)))
if(Kn>D) 
  tol=1e-5; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end

W = zeros(Kn,N);%wij represprent xi  is the set of neighbors of xj and the wight is wij
WW = sparse(1:N,1:N,zeros(1,N),N,N,4*Kn*N);%N*N
IsubW  = sparse(1:N,1:N,ones(1,N),N,N,4*Kn*N);%N*N
for i = 1:N
    z = X(neighborhood(:,i),:) - repmat(X(i,:),Kn,1); % shift ith pt to origin  K*D
    G = z*z';                                         % local covariance  K*K
    G = G + eye(Kn,Kn)* tol * trace(G);                 % regularlization
    W(:,i) = G\ones(Kn,1);                             % solve Gw = 1
    W(:,i) = W(:,i)/sum(W(:,i));                     % normalize
    w = W(:,i);
    j = neighborhood(:,i);
    WW(i,j) = w';
    IsubW(i,j) = IsubW(i,j) - w';
end

%%
% EM iteration
%figure;
PS = [];
while (iter<MaxIter) && (abs(tecr) > ecr) && (sigma2 > 1e-8) 
    %% E-step.
    % Update P
    E_old = E;
    Tx = X + U*C;
    [P1, E] = get_P(Y, Tx, sigma2 ,gamma, a);  
    PS = [PS, P1];
    %DrawArrow2(X,Y,P1);
    P1 = max(P1, minP);
    P1 = max(P1, minP);
    P = sparse(1:N,1:N,P1,N,N,N);
    Q = IsubW'*P*IsubW;
    %update E 
    E = E + eta/2*norm(sqrt(P) * IsubW * Tx,'fro');
    tecr=(E-E_old)/E;
    
    %% M-step.
    % update C by solving linear system
    % update sigma^2 and gamma by tr(v_TPv)/D.tr(P)  and tr(P)/N

    %C = (U'*P*U + eta*sigma2*U'*Q*U)\(U'*P*Y-U'*(P + eta*sigma2*Q)*X);%M*2
    % C = (U'*(P + eta*sigma2*Q)*U) \ (U'*P*Y-U'*(P + eta*sigma2*Q)*X); 
    C = (U'*P*U + eta*sigma2*U'*Q*U)\(U'*(P*Y-(P + eta*sigma2*Q)*X));
    V = Y - X - U*C;
    sigma2 = trace(V'*P*V)/(D*trace(P));
    numcorr = length(find(P > theta));
    gamma=numcorr/N;
    if gamma > 0.95, gamma = 0.95; end
    if gamma < 0.05, gamma = 0.05; end
    iter=iter+1;
end
%%
VecFld.X = X;
VecFld.Y = Y;
VecFld.beta = beta;
VecFld.TX= X + U*C;
VecFld.C=C;
VecFld.ctrl_pts = ctrl_pts;
VecFld.P = diag(P);
VecFld.PS = PS;
VecFld.VFCIndex = find(VecFld.P > theta);

% disp('Removing outliers succesfully completed.');


%%%%%%%%%%%%%%%%%%%%%%%%
function K=con_K(x,y,beta)
% CON_K constructs the kernel K, 
%   where K(i, j) = k(x, y) = exp(-beta*||x-y||^2).

[n, d]=size(x); [m, d]=size(y);

K=repmat(x,[1 1 m])-permute(repmat(y,[1 1 n]),[3 2 1]);
K=squeeze(sum(K.^2,2));
K=-beta * K;
K=exp(K);

%%%%%%%%%%%%%%%%%%%%%%%%
function [P, E]=get_P(Y, Tx, sigma2 ,gamma, a)
% GET_P estimates the posterior probability and part of the energy.

D = size(Y, 2);
temp1 = exp(-sum((Y-Tx).^2,2)/(2*sigma2));
temp2 = (2*pi*sigma2)^(D/2)*(1-gamma)/(gamma*a);
P=temp1./(temp1+temp2);
E=P'*sum((Y-Tx).^2,2)/(2*sigma2)+sum(P)*log(sigma2)*D/2 - log(gamma)*sum(P) - log(1-gamma)*sum(1-P);
