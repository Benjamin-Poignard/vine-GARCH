function [logL, Rt, likelihoods, Qt]=dcc_mvgarch_process(parameters,data,T_in)

data_in = data(1:T_in,:); data = data(T_in+1:end,:);
[t,k]=size(data);
index=1;
H=zeros(size(data));

for i=1:k
    univariateparameters=parameters(index:index+1+1);
    [simulatedata, H(:,i)] = dcc_univariate_simulate(univariateparameters,1,1,data(:,i));
    index=index+1+1+1;
end

stdresid=data./sqrt(H);

stdresid=[ones(1,k);stdresid];
a=parameters(index:index);
b=parameters(index+1:index+1);

Qbar=cov();
Qt=zeros(k,k,t+1);
Qt(:,:,1)=repmat(Qbar,[1 1 1]);
Rt=zeros(k,k,t+1);


logL=0;
likelihoods=zeros(t+1,1);
H=[zeros(1,k);H];
for j=2:t+1
    Qt(:,:,j)=Qbar*(1-a-b);
    Qt(:,:,j)=Qt(:,:,j)+a*(stdresid(j-1,:)'*stdresid(j-1,:));
    Qt(:,:,j)=Qt(:,:,j)+b*Qt(:,:,j-1);
    Rt(:,:,j)=Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
    likelihoods(j)=k*log(2*pi)+sum(log(H(j,:)))+log(det(Rt(:,:,j)))+stdresid(j,:)*inv(Rt(:,:,j))*stdresid(j,:)';
    logL=logL+likelihoods(j);
end;
Rt=Rt(:,:,(2:t+1));
logL=(1/2)*logL;
likelihoods=(1/2)*likelihoods(2:t+1);
