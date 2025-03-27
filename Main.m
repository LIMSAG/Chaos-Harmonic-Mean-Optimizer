% Harmonic Mean Optimizer (HMO) code creater Dr. Pradeep Jangir (pkjmtech@gmail.com)

clear 
close all
clc

nP=40;          % Number of Population

Func_name='F6'; % Name of the test function, range from F1-F23

MaxIt=1000;      % Maximum number of iterations

% Load details of the selected benchmark function
[lb,ub,dim,fobj]=BenchmarkFunctions(Func_name);

[Best_fitness,BestPositions,Convergence_curve] = HMO(nP,MaxIt,lb,ub,dim,fobj);
[Best_fitness1,BestPositions1,Convergence_curve1] = CHMO(nP,MaxIt,lb,ub,dim,fobj);


%% Draw objective space

figure

semilogy(Convergence_curve,'Color','r','LineWidth',4);
hold on
semilogy(Convergence_curve1,'Color','k','LineWidth',4);

title('Convergence curve')
xlabel('Iteration');
ylabel('Best fitness obtained so far');
axis tight
grid off
box on
legend('HMO','CHMO')


