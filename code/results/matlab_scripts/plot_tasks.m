%plot tasks in capability hypercube

close all; clear all;

load('../tasks/TA_normaldist_tasks_0.mat') %loading 500 tasks


show_animation = 0; %1 to see animation of each task appearing

human_c1 = (human_l1 + human_u1)/2; 
human_c2 = (human_l2 + human_u2)/2;
robot_c1 = (robot_l1 + robot_u1)/2;
robot_c2 = (robot_l2 + robot_u2)/2;

num_tasks = 500; %Max = 500
p_num_tasks = p(:, 1:num_tasks); %take the tasks for the number of tasks you want
    
%create plot of task requirements
h = figure(1);
hold on;
plot(human_c1,human_c2,'b*','MarkerSize',12,'linewidth',2,'DisplayName','human capability');
plot(robot_c1,robot_c2,'r*','MarkerSize',12,'linewidth',2,'DisplayName','robot capability');
plot(p_num_tasks(1,:),p_num_tasks(2,:),'k.','DisplayName','task');
xlim([0 1]);
ylim([0 1]);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
title('Tasks','FontSize', 24);
legend;
hold off;


if show_animation == 1
    %animate the plot to plot a point with a 0.1 sec pause
    figure(2);
    hold on;
    h2 = scatter(p_num_tasks(1,1),p_num_tasks(2,1),'k.','DisplayName','task');
    plot(human_c1,human_c2,'b*','MarkerSize',12,'linewidth',2,'DisplayName','human capability');
    plot(robot_c1,robot_c2,'r*','MarkerSize',12,'linewidth',2,'DisplayName','robot capability');
    xlim([0 1]);
    ylim([0 1]);
    xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
    ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
    title('Tasks','FontSize', 24);
    legend;

    for k = 2:num_tasks 
         h2.XData = p_num_tasks(1,1:k); 
         h2.YData = p_num_tasks(2,1:k);

         pause(0.1)
    end

end