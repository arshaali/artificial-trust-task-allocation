%this will load a file
%plot the allocations
%plot the outcomes

%also print success rate, team total reward

%save using save button on plot

close all; clear all;


%load('../atta/atta_caseI_eta50_3.mat')
%load('../random/random_caseI_eta50_3.mat')

load('../atta/atta_caseII_eta50_3.mat')


human_c1 = (human_l1 + human_u1)/2; 
human_c2 = (human_l2 + human_u2)/2;
robot_c1 = (robot_l1 + robot_u1)/2;
robot_c2 = (robot_l2 + robot_u2)/2;


%create plot of task allocation
a = figure(1);
hold on;
plot(human_c1,human_c2,'b*','MarkerSize',14,'linewidth',2,'DisplayName','human capability');
plot(robot_c1,robot_c2,'r*','MarkerSize',14,'linewidth',2,'DisplayName','robot capability');
xlim([0 1]);
ylim([0 1]);
plot(human_p1,human_p2,'b.','MarkerSize',14,'DisplayName','human allocation');
plot(robot_p1,robot_p2,'r.','MarkerSize',14,'DisplayName','robot allocation');
set(gca,'FontSize',24);
%legend;
%legend('Location','South','Orientation','horizontal');
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
title('ATTA','FontSize', 24); %change title to match task allocation method that is being plotted
hold off;


%create plot of task allocation and outcomes
b = figure(2);
hold on;
plot(human_successes(1,:),human_successes(2,:),'b.','linewidth',1,'MarkerSize',14,'MarkerFaceColor','b','DisplayName','Human Success');
plot(human_failures(1,:),human_failures(2,:),'o','linewidth',1,'MarkerSize',6,'MarkerEdgeColor','b','DisplayName','Human Failure');
plot(robot_successes(1,:),robot_successes(2,:),'r.','linewidth',1,'MarkerSize',14,'MarkerFaceColor','r','DisplayName','Robot Success');
plot(robot_failures(1,:),robot_failures(2,:),'o','linewidth',1,'MarkerSize',6,'MarkerEdgeColor','r','DisplayName','Robot Failure');
plot(human_c1,human_c2,'b*','MarkerSize',14,'linewidth',2,'DisplayName','human capability');
plot(robot_c1,robot_c2,'r*','MarkerSize',14,'linewidth',2,'DisplayName','robot capability');
xlim([0 1]);
ylim([0 1]);
set(gca,'FontSize',24);
%legend;
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
title('ATTA','FontSize', 24); %change title to match task allocation method that is being plotted
hold off;

team_perf = (size(human_successes,2) + size(robot_successes,2))/(size(human_successes,2) + size(human_failures,2) + size(robot_successes,2) + size(robot_failures,2));
human_perf = (size(human_successes,2)/(size(human_successes,2) + size(human_failures,2)));
robot_perf = size(robot_successes,2)/(size(robot_successes,2) + size(robot_failures,2));
team_total_reward = total_reward/max_total_reward;
disp('Number of tasks allocated = ')
disp(total_num_tasks)
disp('Team Perf = ')
disp(team_perf)
disp('Human Perf = ')
disp(human_perf)
disp('Robot Perf = ')
disp(robot_perf)
disp('Team Total Reward = ')
disp(team_total_reward)
    