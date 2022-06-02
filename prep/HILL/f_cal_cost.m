function cost = f_cal_cost(cover)
%%Get filter
HF1=[-1,2,-1;2,-4,2;-1,2,-1];
H2 = fspecial('average',[3 3]);
%% Get cost
cover=double(cover);
sizeCover=size(cover);
padsize=max(size(HF1));
coverPadded = padarray(cover, [padsize padsize], 'symmetric');% add padding
R1 = conv2(coverPadded,HF1, 'same');%mirror-padded convolution
W1 = conv2(abs(R1),H2,'same');
if mod(size(HF1, 1), 2) == 0, W1= circshift(W1, [1, 0]); end;
if mod(size(HF1, 2), 2) == 0, W1 = circshift(W1, [0, 1]); end;
W1 = W1(((size(W1, 1)-sizeCover(1))/2)+1:end-((size(W1, 1)-sizeCover(1))/2), ((size(W1, 2)-sizeCover(2))/2)+1:end-((size(W1, 2)-sizeCover(2))/2));
rho=1./(W1+10^(-10));
HW =  fspecial('average',[15 15]);
cost = imfilter(rho, HW ,'symmetric','same');

 