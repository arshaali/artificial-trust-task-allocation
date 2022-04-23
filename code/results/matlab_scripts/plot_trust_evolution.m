%trust evolution

load('../atta/atta_caseII_eta50_3.mat')

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

%evolved trust plot
lambda1 = 0:0.02:1;
lambda2 = lambda1;
[X,Y] = meshgrid(lambda1,lambda2);
F = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(end))
            tp1 = 1;
        elseif (lambda1(i) >= u1(end))
            tp1 = 0;
        else
            tp1 = (u1(end) - lambda1(i))/(u1(end) - l1(end));
        end
            
        if (lambda2(j) <= l2(end))
            tp2 = 1;
        elseif (lambda2(j) >= u2(end))
            tp2 = 0;
        else
            tp2 = (u2(end) - lambda2(j))/(u2(end) - l2(end));
        end
        
        F(i,j) = tp1*tp2;
        
    end
end

%for kH = end
figure(1);
hold on
g = surf(X,Y,F);
alpha 0.5 %transparency of plot
set(g,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = end$','FontSize',30,'interpreter','latex');
hold off


%pre evolved trust plot kH = 0
F2 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= 0)
            tp1 = 1;
        elseif (lambda1(i) >= 1)
            tp1 = 0;
        else
            tp1 = (1 - lambda1(i))/(1 - 0);
        end
            
        if (lambda2(j) <= 0)
            tp2 = 1;
        elseif (lambda2(j) >= 1)
            tp2 = 0;
        else
            tp2 = (1 - lambda2(j))/(1 - 0);
        end
        
        F2(i,j) = tp1*tp2;
        
    end
end

figure(2);
hold on
g2 = surf(X,Y,F2);
alpha 0.5
set(g2,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 0$','FontSize',30,'interpreter','latex');
hold off;

%for kH = 5
F3 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(6))
            tp1 = 1;
        elseif (lambda1(i) >= u1(6))
            tp1 = 0;
        else
            tp1 = (u1(6) - lambda1(i))/(u1(6) - l1(6));
        end
            
        if (lambda2(j) <= l2(6))
            tp2 = 1;
        elseif (lambda2(j) >= u2(6))
            tp2 = 0;
        else
            tp2 = (u2(6) - lambda2(j))/(u2(6) - l2(6));
        end
        
        F3(i,j) = tp1*tp2;
        
    end
end

figure(3);
hold on
g3 = surf(X,Y,F3);
alpha 0.5 %transparency of plot
set(g3,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 5$','FontSize',30,'interpreter','latex');
hold off



%for kH = 10
F4 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(11))
            tp1 = 1;
        elseif (lambda1(i) >= u1(11))
            tp1 = 0;
        else
            tp1 = (u1(11) - lambda1(i))/(u1(11) - l1(11));
        end
            
        if (lambda2(j) <= l2(11))
            tp2 = 1;
        elseif (lambda2(j) >= u2(11))
            tp2 = 0;
        else
            tp2 = (u2(11) - lambda2(j))/(u2(11) - l2(11));
        end
        
        F4(i,j) = tp1*tp2;
        
    end
end

figure(4);
hold on
g4 = surf(X,Y,F4);
alpha 0.5 %transparency of plot
set(g4,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 10$','FontSize',30,'interpreter','latex');
hold off



%for kH = 20
F5 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(21))
            tp1 = 1;
        elseif (lambda1(i) >= u1(21))
            tp1 = 0;
        else
            tp1 = (u1(21) - lambda1(i))/(u1(21) - l1(21));
        end
            
        if (lambda2(j) <= l2(21))
            tp2 = 1;
        elseif (lambda2(j) >= u2(21))
            tp2 = 0;
        else
            tp2 = (u2(21) - lambda2(j))/(u2(21) - l2(21));
        end
        
        F5(i,j) = tp1*tp2;
        
    end
end

figure(5);
hold on
g5 = surf(X,Y,F5);
alpha 0.5 %transparency of plot
set(g5,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 20$','FontSize',30,'interpreter','latex');
hold off


%for kH = 40
F6 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(41))
            tp1 = 1;
        elseif (lambda1(i) >= u1(41))
            tp1 = 0;
        else
            tp1 = (u1(41) - lambda1(i))/(u1(41) - l1(41));
        end
            
        if (lambda2(j) <= l2(41))
            tp2 = 1;
        elseif (lambda2(j) >= u2(41))
            tp2 = 0;
        else
            tp2 = (u2(41) - lambda2(j))/(u2(41) - l2(41));
        end
        
        F6(i,j) = tp1*tp2;
        
    end
end

figure(6);
hold on
g6 = surf(X,Y,F6);
alpha 0.5 %transparency of plot
set(g6,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 40$','FontSize',30,'interpreter','latex');
hold off


%for kH = 100
F7 = zeros(length(lambda1),length(lambda2));
for i = 1:length(lambda1)
    for j = 1:length(lambda2)
        if (lambda1(i) <= l1(101))
            tp1 = 1;
        elseif (lambda1(i) >= u1(101))
            tp1 = 0;
        else
            tp1 = (u1(101) - lambda1(i))/(u1(101) - l1(101));
        end
            
        if (lambda2(j) <= l2(101))
            tp2 = 1;
        elseif (lambda2(j) >= u2(101))
            tp2 = 0;
        else
            tp2 = (u2(101) - lambda2(j))/(u2(101) - l2(101));
        end
        
        F7(i,j) = tp1*tp2;
        
    end
end

figure(7);
hold on
g7 = surf(X,Y,F7);
alpha 0.5 %transparency of plot
set(g7,'edgecolor',[0.5,0.5,0.5])
xlim([0 1]);
ylim([0 1]);
xticks([0 0.5 1]);
yticks([0 0.5 1]);
set(gca,'FontSize',24);
xlabel('$\bar{\lambda}_1$','FontSize', 24,'interpreter','latex');
ylabel('$\bar{\lambda}_2$','FontSize', 24,'interpreter','latex');
zlabel('\tau^H','FontSize',24);
title('$k^H = 100$','FontSize',30,'interpreter','latex');
hold off
