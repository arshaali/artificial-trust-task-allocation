%plot the progression of lower and upper bounds

%print convergence offset, final bounds

close all; clear all;

load('../atta/atta_caseII_eta50_3.mat')


human_c1 = (human_l1 + human_u1)/2; 
human_c2 = (human_l2 + human_u2)/2;
robot_c1 = (robot_l1 + robot_u1)/2;
robot_c2 = (robot_l2 + robot_u2)/2;

%create a plot for the progression of lower and upper bounds over the
%number of tasks assigned to the human
%create an array of the progress in the lower and upper bounds over the
%number of tasks executed by the human
l1 = zeros(1, human_num_tasks + 1);
u1 = zeros(1, human_num_tasks + 1);
l2 = zeros(1, human_num_tasks + 1);
u2 = zeros(1, human_num_tasks + 1);
l1(1) = 0; %lower bounds start at 0
u1(1) = 1; %upper bounds start at 1
l2(1) = 0;
u2(1) = 1;
j = 2;
for i = 1:(length(task_number) - 1) %minus 1
    if task_number(i) ~= task_number(i+1)
        l1(j) = l_1(i);
        u1(j) = u_1(i);
        l2(j) = l_2(i);
        u2(j) = u_2(i);
        j = j + 1;
    end
end
l1(j) = l_1(end);
u1(j) = u_1(end);
l2(j) = l_2(end);
u2(j) = u_2(end);
%l1(i+1) is the lower bound in the first capability dimension after task i
%is executed by the human (e.g., l1(2) is the lower bound after the human has executed 1 task

a = figure(1);
hold on
ntasks = 0:1:human_num_tasks;

plot(ntasks,l1,'b-','linewidth',4,'DisplayName','$\ell_1^H \;\;\;\;$');
plot(ntasks,u1,'b:','linewidth',4,'DisplayName','$u_1^H \;\;\;\;$');
plot(ntasks(1),human_c1,'b*','MarkerSize',14,'linewidth',2,'DisplayName','$\lambda_1^H \;\;\;\;$');
plot(ntasks,l2,'g-','linewidth',4,'DisplayName','$\ell_2^H \;\;\;\;$');
plot(ntasks,u2,'g:','linewidth',4,'DisplayName','$u_2^H \;\;\;\;$');
plot(ntasks(1),human_c2,'g*','MarkerSize',14,'linewidth',2,'DisplayName','$\lambda_2^H$');
xlim([0,ntasks(end)]);
ylim([0 1]);
set(gca,'FontSize',24);
xlabel('Number of Tasks','FontSize', 24);
ylabel('$bel (\lambda_i^H)$','FontSize', 24,'interpreter','latex');
title('Human Capabilities Belief Distribution','FontSize', 24);
%legend('Location','southwest','NumColumns',6,'interpreter','latex')
hold off

disp('True lambda_1 = ')
disp(human_c1)
disp('l_1 = ')
disp(l_1(end))
disp('u_1 = ')
disp(u_1(end))
disp('True lambda_2 = ')
disp(human_c2)
disp('l_2 = ')
disp(l_2(end))
disp('u_2 = ')
disp(u_2(end))
