% Test l1-l2(x_0) regularized logisitic regression

randn('state',0);
rand('state',0);

% % Generate data
% n=200;
% p=20;
% A=[rand(n,p)+ones(n,p);randn(n,p)];
% y=[ones(n,1);-ones(n,1)];
% C=0.1;
% lambda=0.5;
% rho=10;
% 
% 
% if 0 % Launch solver
% tic
% cvx_expert true
% %cvx_precision low
% cvx_begin
%     variables x(p)
%     variable v
%     minimize log_sum_exp(-y.*(A*x+v))+C*norm(x-x_0,2)+lambda*norm(x,1)
% cvx_end
% toc 
% end

% Predictions
% (A*x+z).*y

% Get data
load /Users/gerauddaspremont/Desktop/project/BBCmatlab/topicsMat.mat
load /Users/gerauddaspremont/Desktop/project/BBCmatlab/cameron_labels.mat


% Proces data
n=415041;
A=m(1:n,:);
%A = [sparse(ones(size(A,1),1)), A]; % DEBUG: pas de norme l_1 et l_2 sur la constante...
p=size(A,2);
y=2*labels(1:n)-1;

% Params
x_0=-0.1*rand(p,1);
C=1e-3;
lambda=1e-3;
rho=1e+2;


% Use Newton's methodc
nmax=50;
x=x_0;
ls_alpha=0.1;
ls_beta=0.5;
disp('Newton l1-LogReg starting...');
cvhist=[];
for kit=1:nmax
    % function value
    fv=log(sum(exp(-y.*(A*x)),1))+C*norm(x-x_0,2)^2/2+(lambda/rho)*sum(log(exp(-rho*x)+exp(rho*x)));
    % Compute gradient and Hessian
    z=exp(-y.*(A*x));
    Dy=spdiags(-y,0,length(y),length(y));
    %z = exp(A*x);
    grad=A'*Dy*(z/sum(z))+C*(x-x_0)+...
        lambda*(exp(rho*x)-exp(-rho*x))./(exp(rho*x)+exp(-rho*x));
    ztA=z'*Dy*A;
    H=(A'*Dy*spdiags(z,0,length(z),length(z))*Dy*A)/sum(z);
    H=H-(ztA'*ztA)/sum(z)^2;
    H=H+C*speye(p);
    H=H+lambda*rho*diag((1-(exp(rho*x)-exp(-rho*x)).^2./(exp(rho*x)+exp(-rho*x)).^2));
    % Solve direction
    dx=-H\grad;
    % Line search
    t=1;stop=0;nlsiters=0;
    while stop==0
        newv=log(sum(exp(-y.*(A*(x+t*dx))),1))+C*norm((x+t*dx)-x_0,2)^2/2+(lambda/rho)*sum(log(exp(-rho*(x+t*dx))+exp(rho*(x+t*dx))));
        nlsiters=nlsiters+1;
        if newv < fv + ls_alpha*t*dx'*grad
            stop=1;
        else
            t=t*ls_beta;
        end
        if nlsiters>100 %30
            disp('LS iter limit reached');
            break;
        end
    end
    x=x+t*dx;
    ndec=-grad'*dx;
    fprintf('Iter: %2.0d  Obj: %.4e  Ndec: %.2e\n',kit,fv,ndec);
    cvhist=[cvhist;kit,fv,ndec];
    if ndec<1e-6 break; end
end
% Check results, i^th prediction is sign of (Ax)_i
subplot(2,2,1);hist(y.*(A*x));
set(gca,'FontSize',16);;xlabel('y.*(A*x)');
subplot(2,2,3);semilogy(cvhist(:,1),cvhist(:,3),'LineWidth',3)
set(gca,'FontSize',16);
xlabel('Iteration')
ylabel('Gardien norm')
subplot(2,2,4);semilogy(cvhist(:,1),cvhist(:,2)-min(cvhist(:,2)),'LineWidth',3)
set(gca,'FontSize',16);
xlabel('Iteration')
ylabel('fk-fbest')
