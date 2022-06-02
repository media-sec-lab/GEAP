function generate_cost_ppm(img_path,cost_path,cost_func,payload,ext_img)
% %By: Qin Xinghong
% %Date: 2020.09.24
% %Desc: Generate cost of image. Target file: {'coef':image,'rhoP1",rhoP1,
% 'rhoM1':rhoM1}.
% %Input: img_path--the source folder.
% %       t_file--target file name (including path). If t_file is empty, do
% %             not save the result to any file.
% %       img_count--count of images extracted.
% %Output: f_ret--extracted features.
% %Update: 2020.03.17. Change to rho=(1/beta-1)/lamda to avoid
% computational diffendence. For MiPOD and MGN.

%    img_path='/home/qinxinghong/data/BossBase-1.01-cover';
    if isdir(img_path)==0
        fprintf('The source image directory: %s is not exist!\r\n',img_path);
        return;
    end

    if nargin<5
        ext_img='.ppm';
    end
    f_list=dir([img_path filesep '*' ext_img]); %dir([img_path filesep '*.ppm']);
    len_list=length(f_list);
    if len_list<3
        fprintf('No any file in the source image directory!\r\n');
        return;
    end
    
    d_tar=sprintf('%s%s%s',cost_path,filesep,cost_func);
    if payload>0
        u_p=uint8(100*payload);
        d_tar=sprintf('%s_p%02d',d_tar,u_p);
    end 
    if isdir(d_tar)~=1
        mkdir(d_tar);
    end
    
    addpath(genpath(pwd));
    
    disp('starting generating...');
%     for i = 1:len_list
    parfor i = 1:len_list
        img_name = f_list(i).name;
        f_name = [img_path filesep img_name];
        disp(['Generating the ' num2str(i) ', image: ' img_name '...']);
        generate_cost(img_path, img_name,cost_func,cost_path,payload);
    end
    disp('---------------extracting finished!--------------------');
end

function cost_e=cpv_cost(cover,ver_no,size_spread)
    changes=[1	1	1; -1	-1	-1; 1	1	0; ...
            1	0	1; 1	0	0; -1	-1	0; ...
            -1	0	-1; -1	0	0; 0	1	1; ...
            0	1	0; 0	-1	-1; 0	-1	0; ...
            0	0	1; 0	0	-1; 1	1	-1; ...
            1	-1	1; 1	-1	-1; 1	-1	0; ...
            1	0	-1; -1	1	1; -1	1	-1; ...
            -1	1	0; -1	-1	1; -1	0	1; ...
            0	1	-1; 0	-1	1; 0	0	0];
    cost_e = pv_cost_nephbour(changes,cover,ver_no,size_spread);
end

function cost_e=cal_cost(X,cost_func,Extimate_payload)
    sz=size(X);
    cost_e=zeros(sz);
    for i=1:sz(3)
        switch upper(cost_func)
           case 'HILL'
                cost_e(:,:,i)=f_cal_cost(X(:,:,i));% Obtain the costs for all pixels  
           case 'SUNIWARD'
               cost_e(:,:,i)=cost_suniward(X(:,:,i));
           case 'WOW'
               cost_e(:,:,i)=distortion_wow(X(:,:,i));
    %         case 'MG'%Multiple Gaussian model
    %             cost_e=mg_cost(X,Extimate_payload);
            case 'MIPOD'
                p_change=probability_MiPOD (X(:,:,i), Extimate_payload);
    %             cost_e=log(1./p_change-2);
                cost_e(:,:,i)=log(1./p_change-1);
            case 'MGN'%Multiple Gaussian model of  Image noise.
                p_change=mgn21_prob(X(:,:,i), Extimate_payload);
    %             cost_e=log(1./p_change-2);
                cost_e(:,:,i)=log(1./p_change-1);
            case 'CPV'
                cost_e = cpv_cost(X,33,15);
                break;
           otherwise
        end
    end
end

function FI=fi_hs(X, H_fs, FI_type, Estimate_kernel,Mipod_variance)
% %FI_type: max, min, avr
    len_h=length(H_fs);
    [s1,s2]=size(X);
    FI_s=zeros(len_h,s1,s2);
    
    for i=1:len_h
        FI=fi_noise(X, H_fs{i},Estimate_kernel,Mipod_variance);
        FI_s(i,:,:)=FI;
    end
    
    switch(FI_type)
        case 'max'
            FI=max(FI_s);
        case 'min'
            FI=min(FI_s);
        otherwise
            FI=mean(FI_s);
    end
    FI=FI(1,:,:);
end

function p_change=mgn21_prob(X,Extimate_payload)
    H_s=cell(1,3);
    H_s{1}=[1 -2 1];
    H_s{2}=[1 -2 1]';
    H_s{3}=[-1 2 -1;2 -4 2;-1 2 -1];
    Spread_size=7;

    Estimate_kernel=ones(3,3);
    X=double(X);
    FI=fi_hs(X, H_s, 'max', Estimate_kernel,0);    
    % Compute embedding change probabilities and execute embedding
   lf1=fspecial('average',[Spread_size Spread_size]);

    FI=imfilter(FI,lf1,'symmetric','same');    
    FI =FI(:)';

    % Ternary embedding change probabilities
    beta = TernaryProbs(FI,Extimate_payload);
    p_change=reshape(beta,size(X));

end

function [beta] = TernaryProbs(FI,alpha)

    load('/home/qinxinghong/Code/MG/ixlnx3.mat');

    % Absolute payload in nats
    payload = alpha * length(FI) * log(2);

    % Initial search interval for lambda
    [L, R] = deal (10^3, 10^6);

    fL = h_tern(1./invxlnx3_fast(L*FI,ixlnx3)) - payload;
    fR = h_tern(1./invxlnx3_fast(R*FI,ixlnx3)) - payload;
    % If the range [L,R] does not cover alpha enlarge the search interval
    while fL*fR > 0
        if fL > 0
            R = 2*R;
            fR = h_tern(1./invxlnx3_fast(R*FI,ixlnx3)) - payload;
        else
            L = L/2;
            fL = h_tern(1./invxlnx3_fast(L*FI,ixlnx3)) - payload;
        end
    end

    % Search for the labmda in the specified interval
    [i, fM, TM] = deal(0, 1, zeros(60,2));
    while (abs(fM)>0.0001 && i<60)
        M = (L+R)/2;
        fM = h_tern(1./invxlnx3_fast(M*FI,ixlnx3)) - payload;
        if fL*fM < 0, R = M; fR = fM;
        else          L = M; fL = fM; end
        i = i + 1;
        TM(i,:) = [fM,M];
    end
    if (i==60)
        M = TM(find(abs(TM(:,1)) == min(abs(TM(:,1))),1,'first'),2);
    end
    % Compute beta using the found lambda
    beta = 1./invxlnx3_fast(M*FI,ixlnx3);

end

% Fast solver of y = x*log(x-2) paralellized over all pixels
function x = invxlnx3_fast(y,f)

    i_large = y>1000;
    i_small = y<=1000;

    iyL = floor(y(i_small)/0.01)+1;
    iyR = iyL + 1;
    iyR(iyR>100001) = 100001;

    x = zeros(size(y));
    x(i_small) = f(iyL) + (y(i_small)-(iyL-1)*0.01).*(f(iyR)-f(iyL));

    z = y(i_large)./log(y(i_large)-2);
    for j = 1 : 20
        z = y(i_large)./log(z-2);
    end
    x(i_large) = z;

end

% Ternary entropy function expressed in nats
function Ht = h_tern(Probs)

    p0 = 1-2*Probs;
    P = [p0(:);Probs(:);Probs(:)];
    H = -(P .* log(P));
    H((P<eps)) = 0;
    Ht = nansum(H);

end

function EstimatedVariance = VarianceEstimation(Image,Kernel)

% Kernel = ones(3,3);
% Local sums
%x1 = conv2(Image, Kernel, 'same');
x1=imfilter(Image,Kernel,'symmetric','same');
% Local quadratic sums
%x2 = conv2(Image.^2, Kernel, 'same');
x2=imfilter(Image.^2,Kernel,'symmetric','same');
% Number of matrix elements in each square region
%R = conv2(ones(size(Image)), Kernel, 'same');
R = imfilter(ones(size(Image)), Kernel,'symmetric', 'same');
% Local variance
EstimatedVariance = x2./R-(x1./R).^2;

end

function EstimatedVariance = VarianceEstimationDCT2D(Image, BlockSize, Degree)
% verifying the integrity of input arguments
if ~mod(BlockSize,2)
    error('The block dimensions should be odd!!');
end
if (Degree > BlockSize)
    error('Number of basis vectors exceeds block dimension!!');
end

% number of parameters per block
q = Degree*(Degree+1)/2;

% Build G matirx
BaseMat = zeros(BlockSize);BaseMat(1,1) = 1;
G = zeros(BlockSize^2,q);
k = 1;
for xShift = 1 : Degree
    for yShift = 1 : (Degree - xShift + 1)
        G(:,k) = reshape(idct2(circshift(BaseMat,[xShift-1 yShift-1])),BlockSize^2,1);
        k=k+1;
    end
end

% Estimate the variance
PadSize = floor(BlockSize/2*[1 1]);
I2C = im2col(padarray(Image,PadSize,'symmetric'),BlockSize*[1 1]);
PGorth = eye(BlockSize^2) - (G*((G'*G)\G'));
EstimatedVariance = reshape(sum(( PGorth * I2C ).^2)/(BlockSize^2 - q),size(Image));
end

function cost = f_cal_cost(cover)
%%Get filter
    HF1=[-1,2,-1;2,-4,2;-1,2,-1];
    H2 = fspecial('average',[3 3]);
    % % Get cost
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
end

function rho = distortion_wow( cover)

% % Get 2D wavelet filters - Daubechies 8
% 1D high pass decomposition filter
hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
        -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
% 1D low pass decomposition filter
lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);
% construction of 2D wavelet filters
F{1} = lpdf'*hpdf;
F{2} = hpdf'*lpdf;
F{3} = hpdf'*hpdf;

% % Get embedding costs
% inicialization
cover = double(cover);
p = -1;%params.p;
wetCost = 10^10;
sizeCover = size(cover);

% add padding
padSize = max([size(F{1})'; size(F{2})'; size(F{3})']);
coverPadded = padarray(cover, [padSize padSize], 'symmetric');

% compute directional residual and suitability \xi for each filter
xi = cell(3, 1);
for fIndex = 1:3
    % compute residual
    R = conv2(coverPadded, F{fIndex}, 'same');
       
    % compute suitability
    xi{fIndex} = conv2(abs(R), rot90(abs(F{fIndex}), 2), 'same');
    % correct the suitability shift if filter size is even
    if mod(size(F{fIndex}, 1), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [1, 0]); end;
    if mod(size(F{fIndex}, 2), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [0, 1]); end;
    % remove padding
    xi{fIndex} = xi{fIndex}(((size(xi{fIndex}, 1)-sizeCover(1))/2)+1:end-((size(xi{fIndex}, 1)-sizeCover(1))/2), ((size(xi{fIndex}, 2)-sizeCover(2))/2)+1:end-((size(xi{fIndex}, 2)-sizeCover(2))/2));
end

% compute embedding costs \rho
rho = ( (xi{1}.^p) + (xi{2}.^p) + (xi{3}.^p) ) .^ (-1/p);

% adjust embedding costs
rho(rho > wetCost) = wetCost; % threshold on the costs
rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost

end

function cost_ret=cost_suniward(cover)
    sgm = 1;

    % % Get 2D wavelet filters - Daubechies 8
    % 1D high pass decomposition filter
    hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
            -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
    % 1D low pass decomposition filter
    lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);
    % construction of 2D wavelet filters
    F{1} = lpdf'*hpdf;
    F{2} = hpdf'*lpdf;
    F{3} = hpdf'*hpdf;

    % % Get embedding costs
    % inicialization
    if ischar(cover)==1
        cover = double(imread(cover));
    else
        cover=double(cover);
    end

    wetCost = 10^10;
    [k,l] = size(cover);

    % add padding
    padSize = max([size(F{1})'; size(F{2})'; size(F{3})']);
    coverPadded = padarray(cover, [padSize padSize], 'symmetric');

    xi = cell(3, 1);
    for fIndex = 1:3
        % compute residual
        R = conv2(coverPadded, F{fIndex}, 'same');
        % compute suitability
        xi{fIndex} = conv2(1./(abs(R)+sgm), rot90(abs(F{fIndex}), 2), 'same');
        % correct the suitability shift if filter size is even
        if mod(size(F{fIndex}, 1), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [1, 0]); end;
        if mod(size(F{fIndex}, 2), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [0, 1]); end;
        % remove padding
        xi{fIndex} = xi{fIndex}(((size(xi{fIndex}, 1)-k)/2)+1:end-((size(xi{fIndex}, 1)-k)/2), ((size(xi{fIndex}, 2)-l)/2)+1:end-((size(xi{fIndex}, 2)-l)/2));
    end

    % compute embedding costs \rho
    rho = xi{1} + xi{2} + xi{3};

    % adjust embedding costs
    rho(rho > wetCost) = wetCost; % threshold on the costs
    rho(isnan(rho)) = wetCost; % if all xi{} are zero threshold the cost
    
    cost_ret=rho;
end

function generate_cost(cover_path,img_name,cost_func,cost_path,payload)
    img_file=[cover_path filesep img_name];
    I=imread(img_file);
    X=double(I);
    
    cost_e=cal_cost(X,cost_func,payload);
    if strcmp(cost_func, 'CPV')==0
        wet_code=10^10;
        cost_e(isnan(cost_e))=wet_code;
        rhoP1=cost_e;
        rhoM1=cost_e;
        rhoP1(I==255)=wet_code;
        rhoM1(I==0)=wet_code;
    end
    
    f_tar=sprintf('%s%s%s',cost_path,filesep,cost_func);
    [path_i,name_i,ext_i] = fileparts(img_name);
    if payload>0
        f_tar=sprintf('%s_p%02d%s%s.mat',f_tar,uint8(100*payload),filesep,name_i);
    else
        f_tar=sprintf('%s%s%s.mat',f_tar,filesep,name_i);
    end

    if strcmp(cost_func, 'CPV')==0
        coef=X;
        save (f_tar, 'coef','rhoP1','rhoM1','-v6');
    else
        rhoCPV=cost_e;
        save (f_tar, 'rhoCPV','-v6');
    end
    
end

function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

    l3 = 1e+3;
    m3 = double(message_length + 1);
    iterations = 0;
    while m3 > message_length
        l3 = l3 * 2;
        pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
        pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
        m3 = ternary_entropyf(pP1, pM1);
        iterations = iterations + 1;
        if (iterations > 10)
            lambda = l3;
            return;
        end
    end        

    l1 = 0; 
    m1 = double(n);        
    lambda = 0;
    iterations = 0;
    alpha = double(message_length)/n;
    % limit search to 30 iterations
    % and require that relative payload embedded is roughly within 1/1000 of the required relative payload        
    while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<300)
        lambda = l1+(l3-l1)/2; 
        pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
        pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
        m2 = ternary_entropyf(pP1, pM1);
        if m2 < message_length
            l3 = lambda;
            m3 = m2;
        else
            l1 = lambda;
            m1 = m2;
        end
        iterations = iterations + 1;

    end
%         disp(iterations);
end
    
function Ht = ternary_entropyf(pP1, pM1)
    p0 = 1-pP1-pM1;
    P = [p0(:); pP1(:); pM1(:)];
    H = -((P).*log2(P));
    H((P<eps) | (P > 1-eps)) = 0;
    Ht = sum(H);
end
