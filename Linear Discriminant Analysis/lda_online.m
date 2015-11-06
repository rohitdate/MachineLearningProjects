%Please run the code as follow:
%--> lda_online('dataset1.csv',200,0.01)
function lda_online(file,num_iterations,learning_rate)
ds1=dataset('file',file,'delimiter',',');
ds1size=size(ds1);
num_attr=ds1size(2)-1;

weights_rand=rand(1,num_attr);
weights=weights_rand;

x1=zeros(1,100);
x2=zeros(1,100);
x3=zeros(1,100);

error=zeros(1,100);

avg_error=zeros(1,num_iterations);
for i=1:num_iterations              %the biggest for loop until convergence
    ch_weights=zeros(1,num_attr);
    for j=1:ds1size(1)              % it is the number of rows
        o=0;
        for m=1:num_attr                    %it is the num of attributes
           temp=dataset2cell(ds1(j,m));
           temp=cell2mat(temp(2));
           o=o+weights(m)*temp;
           
        end
        
        %disp('predicted value');
        
        y=1./(1+exp(-o));
        actual=dataset2cell(ds1(j,ds1size(2))); 
        
        actual=cell2mat(actual(2));
        error(j)=abs(actual-y);
        avg_error(i)=sum(error)/100;
        %break
        for m=1:num_attr              % number of attributes
            temp1=dataset2cell(ds1(j,ds1size(2)));
            temp1=cell2mat(temp1(2));
            temp2=dataset2cell(ds1(j,m));
            temp2=cell2mat(temp2(2));
            ch_weights(m)=ch_weights(m)+((temp1-y)*temp2); 
            if(m==1)
               x1(1,j)= (temp1-y);
            end
            if(m==2)
               x2(1,j)= (temp1-y);
            end
            if(m==3)
               x3(1,j)= (temp1-y);
            end           
        end
        
        for k=1:num_attr
            weights(k)=weights(k)+(learning_rate*ch_weights(k)); 
        end
    end
    
    %disp('change weights');
    for m=1:num_attr
           ch_weights(m);
    end
    %disp('weights')
    for m=1:num_attr
           weights(m);
    end
    %break
end
l=(1:1:num_iterations);
%scatter(l,x3)
plot(l,avg_error);
xlabel('Number of iterations');
ylabel('error');