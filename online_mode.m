 clc
clear all
%close all
%%%%%%%%%%%%%%%%%%%%%%%%%%% series parallel mode with noise
%%%%%%%%%%%%%%%%%%%%%%%%%% for identifier %%%%%%%%%%%%%%%%%%%%%%%
w1=0.1*rand(20,1);
w2=0.1*rand(20,1);
w3=0.1*rand(20,1);

w5=0.1*rand(20,1);
w7=0.1*rand(20,1);   %o/p weights
wh=0.1*rand(20,1);
wo=0.1*rand;
noise=rand;

alpha=0.01;
eta=0.01; %eta=0.001 for fixed rate; and  0.007 adaptive in all NN
y(1)=rand;
y(2)=rand;
y(3)=rand;
T=1;
yn(1)=rand;
yn(2)=rand;
yn(3)=rand;
n=46000;  % n=96000
MSE=rand;

%for i=1:20  %1:10
    for k=3:n % for k=2:10000

   %%%%%%%%%%%%%%%% external input %%%%%%%%%%%%%%%%%%%%%%
       r(k)=sin(2*pi*k/25);
       
       
       
       %%%%%%%%%%%%%%%%% plant %%%%%%%%%%%%
   y(k+1)=(5*y(k)*y(k-1))/(1+y(k)^2+y(k-1)^2+y(k-2)^2)+r(k)+0.8*r(k-1);
    
    %%%%%%%%%%%%%%%%%%%%%% for identifier %%%%%%%%%%%%%%%%%%%
    net1=tansig(y(k)*w1+y(k-1)*w2+y(k-2)*w3-wh); % 20x1 vector
    yn(k+1)=purelin(net1'*w7-wo)+r(k)+0.8*r(k-1);
    
    %%%%%%%%%%%%%%%%%%%%% error %%%%%%%%%%%%%%%%%%%%%%
    ei(k+1)=y(k+1)-yn(k+1);
    %MSE=MSE+0.5*(ei(k+1))^2;
    MSE(k+1)=0.5*(ei(k+1))^2;
    %%%%%%%%%%%%%%%%%%%%%% updation of identifier weights %%%%%%%%%%%%%%%%
    a=dpurelin(net1'*w7-wo,yn(k+1))*ei(k+1); %gradient of o/p. its a scalar
    b=dtansig(y(k)*w1+y(k-1)*w2+y(k-2)*w3-wh,net1); %derivative of hidden neuron AF, vector
    delw1=eta*y(k)*a*b.*w7;
    delw2=eta*y(k-1)*a*b.*w7;
   delw3=eta*y(k-2)*a*b.*w7;
    
    delw7=eta*net1*a;
    delwh=eta*(-1)*a*b.*w7;
    delwo=eta*(-1)*a;
    w1=w1+delw1;
    w2=w2+delw2;
    w3=w3+delw3;
   
   
   
   
    w7=w7+delw7;
    wo=wo+delwo;
    wh=wh+delwh;
    end
    avg=sum(MSE)/n
%     AMSE(i)=sum(MSE)/k;
%    MSE=0;
% end

figure
plot(yn,'r:')
hold on
plot(y,'b')
figure
plot(MSE)