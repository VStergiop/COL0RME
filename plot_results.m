clear all;

load K700_SNR10

L = size(a.cel0{1},1);


% img1 = GT
img1=a.gt;
img1=img1./max(img1,[],'all');
img1=img1.^.5;

% img2 = raw data
img2=imresize(a.raw,[L,L]);
img2=img2./max(img2,[],'all');

% img3 = l1
l1_reg = 5;
img3=a.l1{l1_reg}/3;

% img4 = TV
tv_reg = 1;
img4 = a.tv{tv_reg}/5;

% img5 = cel0, 0s initialization
cel0_reg = 2;
img5=a.cel0{cel0_reg};

% img6 = cel0, with l1 regularizer
cel0_l1_reg = cel0_reg;
img6=a.cel0_initl1{cel0_l1_reg};




figure(3);
clf
subplot 331
RGB=cat(3,img1,zeros(L),img2);
imagesc(RGB);
axis image;
colormap('hot')
title('Ground Truth and Blurred Raw Data')
subplot 334
imagesc(img4);
axis image;
colormap('hot')
title(['TV, JI:' num2str(a.tvJaccAll{tv_reg}), ', FP:' num2str(a.tvJaccFP{tv_reg})])
subplot 335
imagesc(img3);
axis image;
colormap('hot')
title(['L1, JI:' num2str(a.l1JaccAll{l1_reg}), ', FP:' num2str(a.l1JaccFP{l1_reg})])
subplot 337
imagesc(img5);
axis image;
colormap('hot')
title(['CEL0 (0s), JI:' num2str(a.cel0JaccAll{cel0_reg}), ', FP:' num2str(a.cel0JaccFP{cel0_reg})])
subplot 338
imagesc(img6);
axis image;
colormap('hot')
title(['CEL0 (l1), JI:' num2str(a.cel0JaccAll_initl1{cel0_l1_reg}), ', FP:' num2str(a.cel0JaccFP_initl1{cel0_l1_reg})])

subplot 133
RGB=cat(3,img1,img3,img3);
imagesc(RGB);
% hold on;
% [row,col] = find(a.cel0{cel0_reg});
% plot(col,row,'.','Color','green')
% set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
hold on;
[row,col] = find(a.cel0_initl1{cel0_l1_reg});
plot(col,row,'.','Color','green')
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
axis image;
colormap('hot')
set(gcf,'name','GT-EstSupp')

%%
% figure(4);
% clf
% subplot 221
% imagesc(Vassi.angle2.Raw);
% axis image;
% colorbar
% colormap('hot')
% subplot 223
% imagesc(Vassi.angle2.Celo);
% axis image;
% colorbar
% colormap('hot')
% subplot 122
% RGB=cat(3,img3,img4,img3);
% imagesc(RGB);
% axis image;
% colormap('hot')
% set(gcf,'name','Angle2')
% 
% %%
% k=imresize(Vassi.angle3.Raw,[512,512]);
% x=180:300;
% crv1=k(433,x);
% k=Vassi.angle3.Celo;
% crv2=k(433,x);
% 
% figure(5)
% h=plot(x,crv1-min(crv1));
% hold on
% h=plot(x,crv2)
% hold off
% datatip(h,219,4469000);
% datatip(h,247,1398000);
% datatip(h,259,532200);
% datatip(h,273,260600);
% datatip(h,282,34670);
% 
% 
