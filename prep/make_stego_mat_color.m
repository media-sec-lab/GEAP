function make_stego_mat_color(cover_dir,target_dir,cost_func,cost_dir ...
    ,payload_rate,is_cmd,is_stc,result_file,ext_img)
% %By: Qin Xinghong
% %Date: 2020.09.24.
% %Update: 2019.12.18. Add CMD profile and result records.
% %Update: 2020.03.18. Change to rho=(1/beta-1)/lamda to avoid
% %Update: 2020.07.15. Randomly select the start sub-image for CMD.
% computational diffendence. For MiPOD and MGN.

    if strcmp(cost_func,'CPV')==1 && is_cmd==1
        fprintf('Please use test_pv_cost_ex.m to create CMD version CPV stego images!');
        return;
    end
    
    if nargin<9
        ext_img='.ppm';
    end
    f_list=dir([cover_dir filesep '*' ext_img]); %dir([cover_dir filesep '*.ppm']);
    len_f=length(f_list);
    if len_f<=0 
        disp('There is no any file in the cover path!');
        return;
    end
    u_p=uint8(100*payload_rate);
    t_dir=sprintf('%s%s%sS%1d_p%02d', target_dir,filesep,cost_func, is_stc, u_p);
    if is_cmd==1
        t_dir=sprintf('%s%s%sS%1d_CMD_p%02d', target_dir,filesep,cost_func, is_stc, u_p);
    end
    if is_cmd>=2  % randomly select the start sub-image.
        t_dir=sprintf('%s%s%sS%1d_RM%1d_p%02d', target_dir,filesep,cost_func, is_stc, is_cmd, u_p);
    end
    if isdir(t_dir)==0
        mkdir(t_dir);
    end

    addpath(genpath(pwd));

    emb_results=zeros(len_f,3);
%     for i=1:len_f
    parfor i=1:len_f
        f_name=f_list(i).name;
        [stc_code,diff_cs,start_block]= embed_data(cover_dir,f_name ...
            , target_dir,cost_func,cost_dir,payload_rate,is_cmd,is_stc);
        emb_results(i,:)=[stc_code diff_cs start_block];
    end
    
    if length(result_file)>3
        f_id=fopen(result_file,'w+');
        fprintf(f_id,'make_stego_mat.m\n');
        fprintf(f_id,'cover:,%s\ntarget:,%s\n',cover_dir,target_dir);
        fprintf(f_id,'cost:,%s,payload:,%.2f,CMD,%d,STC:,%d\n',cost_func,payload_rate,is_cmd,is_stc);
        fprintf(f_id,'name,stc,difference,start\n');
        for i=1:len_f
            fprintf(f_id,'%s,%d,%d,%d\n',f_list(i).name,emb_results(i,1),emb_results(i,2),emb_results(i,3));
        end
        fclose(f_id);
    end
    disp('--------Finished---------');
end

function [stc_code, diff_cs, start_block] = embed_data(cover_dir,img_name ...
    , target_dir,cost_func,cost_dir,payload_rate,is_cmd,is_stc)
    if is_cmd==1
        start_block=1;
        [stc_code,diff_cs]= embed_cmd(cover_dir,img_name, target_dir ...
            ,cost_func,cost_dir,payload_rate,is_stc);
    else
        start_block=1;
        [stc_code,diff_cs]= embed_basic(cover_dir,img_name, target_dir ...
            ,cost_func,cost_dir,payload_rate,is_stc);
    end
end

function [stc_code, diff_cs] = embed_basic(cover_dir,img_name ...
    , target_dir,cost_func,cost_dir,payload_rate,is_stc)
    fprintf('[%s-p%.2f]%s is being peocessed...\n',cost_func,payload_rate, img_name);
    f_name=[cover_dir filesep img_name];
    Cover=imread(f_name);
    X=double(Cover);
    [dir_i,name_i,ext_i]=fileparts(img_name);
    f_name=[cost_dir filesep name_i '.mat'];
    Cover=load(f_name);
    stc_code=is_stc;
    if strcmp(cost_func,'CPV')==0
        rhoP1=Cover.rhoP1;
        rhoM1=Cover.rhoM1;
        Y=X;
        for i=1:3
            if is_stc==0
                [Y_i, Px]=sim_embedding(X(:,:,i),rhoP1(:,:,i),rhoM1(:,:,i),payload_rate);
                stc_code = 0;
            else
                H=10;
                [Y_i, stc_code]=stc_embedding(X(:,:,i),rhoP1(:,:,i),rhoM1(:,:,i),payload_rate,H);
            end
            Y(:,:,i)=Y_i;
        end
    else
        cost_27=Cover.rhoCPV;
        [Y,dis]=apv_embed(X,payload_rate,cost_27,is_stc,10);
    end
    
    u_p=uint8(100*payload_rate);
    t_file=sprintf('%s%s%sS%1d_p%02d%s%s', ...
        target_dir,filesep,cost_func, is_stc, u_p, filesep, img_name);
%     coefC=X;
%     probC=Px;
%     coefS=Y;
%     [rP1,rM1]=cal_cost(Y,cost_func,payload_rate);
%     Py=get_prob(rP1,rM1,payload_rate);
%     probS=Py;
%     save(t_file,'coefC','coefS','probC','probS', '-v6');
    imwrite(uint8(Y), t_file);
    diff_cs=sum(uint8(X(:))~=uint8(Y(:)));
end

function [stc_code, diff_cs] = embed_cmd(cover_dir,img_name, target_dir ...
    ,cost_func,cost_dir,payload_rate,is_stc)
    fprintf('[%s-%.2fp-CMD]%s is being peocessed...\n',cost_func,payload_rate, img_name);
    stc_code=is_stc;
    f_name=[cover_dir filesep img_name];
    Cover=imread(f_name);
    X=double(Cover);
    Y=X;
    [dir_i,name_i,ext_i]=fileparts(img_name);
    f_name=[cost_dir filesep name_i '.mat'];
    Cover=load(f_name);
    if strcmp(cost_func,'CPV')==0
        rhoP1=Cover.rhoP1;
        rhoM1=Cover.rhoM1;
%         probC=get_prob(rhoP1,rhoM1,payload_rate);
        X=int32(X);
        Y=X;
        idx_rc=[1 1;
                1,2;
                2,2;
                2,1]; %start indexes of blocks.
        h_cmd=[0 1 0;
               1 0 1;
               0 1 0];
        H=10;
        for idx_b=1:4
            idx_r=idx_rc(idx_b,1);
            idx_c=idx_rc(idx_b,2);
            X0=X(idx_r:2:end,idx_c:2:end);
            if idx_b>1 %re-compute cost
                [rhoP1,rhoM1]=cal_cost(Y,cost_func,payload_rate);
                D=double(Y)-double(X);
                D=imfilter(D,h_cmd,'symmetric','same');
                rhoP1(D>=1)=rhoP1(D>=1)/10;
                rhoM1(D<=-1)=rhoM1(D<=-1)/10;
            end
            rP1=rhoP1(idx_r:2:end,idx_c:2:end);
            rM1=rhoM1(idx_r:2:end,idx_c:2:end);
            if stc_code==0
                [Y0, Px]=sim_embedding(X0,rP1,rM1,payload_rate);
            else
                [Y0, stc_code]=stc_embedding(X0,rP1,rM1,payload_rate,H);
            end
            Y(idx_r:2:end,idx_c:2:end)=double(Y0);
        end
    else  % CPV. Please use 'test_pv_cost_ex.m'.
        
    end
    
    u_p=uint8(100*payload_rate);
    t_file=sprintf('%s%s%sS%1d_CMD_p%02d%s%s', ...
        target_dir,filesep,cost_func, is_stc, u_p, filesep, img_name);
%     coefC=X;
%     coefS=Y;
% %     [rP1,rM1]=cal_cost(X,cost_func,payload_rate);
% %     probC=get_prob(rP1,rM1,payload_rate);
%     [rP1,rM1]=cal_cost(Y,cost_func,payload_rate);
%     probS=get_prob(rP1,rM1,payload_rate);
%     save(t_file,'coefC','coefS','probC','probS', '-v6');
    imwrite(uint8(Y),t_file);
    diff_cs=sum(int32(X(:))~=int32(Y(:)));
end

function [stc_code, diff_cs, start_block] = embed_rmd(cover_dir,mat_name, target_dir,cost_func,payload_rate,is_stc,is_cmd)
    fprintf('[%s-%.2fp-RMD]%s is being peocessed...\n',cost_func,payload_rate, mat_name);
    stc_code=is_stc;
    f_name=[cover_dir filesep mat_name];
    Cover=load(f_name);
    X=Cover.coef;
    rhoP1=Cover.rhoP1;
    rhoM1=Cover.rhoM1;
    probC=get_prob(rhoP1,rhoM1,payload_rate);
    X=int32(X);
    Y=X;
    idx_rc=[1 1;
            1,2;
            2,2;
            2,1]; %start indexes of blocks.
    h_cmd=[0 1 0;
           1 0 1;
           0 1 0];
    H=10;
    idx_b=randperm(4,1); % the No. of start sub-image.
    start_block=idx_b;
    fprintf('Image: %s, start No.: %d...\n',mat_name,idx_b);
    for idx_i=1:4
        idx_r=idx_rc(idx_b,1);
        idx_c=idx_rc(idx_b,2);
        X0=X(idx_r:2:end,idx_c:2:end);
        if idx_i>1 %re-compute cost
            if is_cmd==2 % re-compute costs.
                [rhoP1,rhoM1]=cal_cost(Y,cost_func,payload_rate);
            end
            D=double(Y)-double(X);
            D=imfilter(D,h_cmd,'symmetric','same');
            rhoP1(D>=1)=rhoP1(D>=1)/10;
            rhoM1(D<=-1)=rhoM1(D<=-1)/10;
        end
        rP1=rhoP1(idx_r:2:end,idx_c:2:end);
        rM1=rhoM1(idx_r:2:end,idx_c:2:end);
        if stc_code==0
            [Y0, Px]=sim_embedding(X0,rP1,rM1,payload_rate);
        else
            [Y0, stc_code]=stc_embedding(X0,rP1,rM1,payload_rate,H);
        end
        Y(idx_r:2:end,idx_c:2:end)=double(Y0);
        
        idx_b=idx_b+1;
        if idx_b>4
            idx_b=1;
        end
    end
    
    u_p=uint8(100*payload_rate);
    t_file=sprintf('%s%s%sS%1d_RM%1d_p%02d%s%s', ...
        target_dir,filesep,cost_func, is_stc, is_cmd, u_p, filesep, mat_name);
    coefC=X;
    coefS=Y;
%     [rP1,rM1]=cal_cost(X,cost_func,payload_rate);
%     probC=get_prob(rP1,rM1,payload_rate);
    [rP1,rM1]=cal_cost(Y,cost_func,payload_rate);
    probS=get_prob(rP1,rM1,payload_rate);
    save(t_file,'coefC','coefS','probC','probS', '-v6');
    diff_cs=sum(int32(X(:))~=int32(Y(:)));
end

function [Y, P]=sim_embedding(X,rhoP1,rhoM1,payload_rate)
% Embedding simulator simulates the embedding made by the best possible ternary coding method (it embeds on the entropy bound). 
% This can be achieved in practice using "Multi-layered  syndrome-trellis codes" (ML STC) that are asymptotically aproaching the bound.
    n = numel(X);   
    m=n*payload_rate;
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    Z=(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeP1 = (exp(-lambda .* rhoP1))./Z;
    pChangeM1 = (exp(-lambda .* rhoM1))./Z;
    P=pChangeP1+pChangeM1;
    seed=sum(100*clock);
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
    randChange = rand(size(X));
    Y = double(X);
    Y(randChange < pChangeP1) = Y(randChange < pChangeP1) + 1;
    Y(randChange >= 1-pChangeM1) = Y(randChange >= 1-pChangeM1) - 1;
    
    Y(Y<0)=1;
    Y(Y>255)=254;
    
    Y=uint8(Y);
end

function P=get_prob(rhoP1,rhoM1,payload_rate)
    n = numel(rhoP1);   
    m=n*payload_rate;
    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    zP=exp(-lambda .* rhoP1);
    zM=exp(-lambda .* rhoM1);
    Z=(1 + zP + zM);
    pChangeP1 = zP./Z;
    pChangeM1 = zM./Z;
    P=pChangeP1+pChangeM1;
end

function [Y,stc_code]=stc_embedding(X,rhoP1,rhoM1,payload_rate,H)
    stc_code=0;
    [r,t]=size(X);
    n=r*t;
    m =floor(n*payload_rate); 
    msg = uint8(rand(1,m));        %Generating a random secert message

    costs = zeros(3, n, 'single'); % Assign costs for +1 and -1
    costs(1,:)=reshape(rhoM1,1,n);
    %costs(2,:)=0;
    costs(3,:)=reshape(rhoP1,1,n);

    cover1=int32(reshape(X,1,n));
    n_times=0;
    while n_times<3
        try
            [distortion, stegoA, n_msg_bits ] = stc_pm1_pls_embed(cover1, costs, msg,H);%embedding message
             extr_msg = stc_ml_extract(stegoA, n_msg_bits, H);
             if all(extr_msg~=msg)
                 fprintf('Message was embedded and extract correctly\n');     
             end
            Y = reshape(stegoA,r,t);
            stc_code=1;
            break;
        catch ex_stc
            fprintf('[STC ERR]Try %d time: %s...\r\n',n_times,ex_stc.message);
            msg = uint8(rand(1,m));        %Generating a random secert message
        end
        n_times=n_times+1;
    end
    
    if stc_code==0
        [Y, P]=sim_embedding(X,rhoP1,rhoM1,payload_rate*1.05); % increase 0.05 times payload.
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

function [rhoP1,rhoM1]=cal_cost(X,cost_func,Extimate_payload)
    sz=size(X);
    rho_0=zeros(sz);
    for i=1:sz(3)
        switch upper(cost_func)
           case 'HILL'
                cost_e=f_cal_cost(X(:,:,i));% Obtain the costs for all pixels  
           case 'SUNIWARD'
               cost_e=cost_suniward(X(:,:,i));
           case 'WOW'
               cost_e=distortion_wow(X(:,:,i));
    %         case 'MG'%Multiple Gaussian model
    %             cost_e=mg_cost(X,Extimate_payload);
            case 'MIPOD'
                p_change=probability_MiPOD (X(:,:,i), Extimate_payload);
                cost_e=log(1./p_change-1);
            case 'MGN'%Multiple Gaussian model of  Image noise.
                p_change=mgn21_prob(X(:,:,i), Extimate_payload);
                cost_e=log(1./p_change-1);
           otherwise
        end
        rho_0(:,:,i)=cost_e;
    end
    
    wet_code=10^10;
    rho_0(isnan(rho_0))=wet_code;
    rhoP1=rho_0;
    rhoM1=rho_0;
    rhoP1(X==255)=wet_code;
    rhoM1(X==0)=wet_code;
    
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

function COST=mgn_cost(X,Extimate_payload)
    H_filter=[-1 2 -1;2 -4 2;-1 2 -1];
    Spread_size=11;
    Mipod_variance=0;
    
    [s1,s2]=size(H_filter);
    F_fi=sum(H_filter(:).*H_filter(:));
    F_fi=F_fi*F_fi/(s1*s2)^2;%(sum(aij^2))^2/(r*s)^2

    %N_C=conv2(Cover,H_filter,'same');
    N_C=imfilter(X,H_filter,'symmetric','same');

    % Compute Variance and do the flooring for numerical stability
    %Variance = VarianceEstimation(Cover);
    if Mipod_variance==1
        Variance= VarianceEstimationDCT2D(N_C, 9, 9);
    else
        Estimate_kernel=ones(3,3);
        Variance = VarianceEstimation(N_C,Estimate_kernel);
    end
    Variance(Variance< 0.01) = 0.01;

    % Compute Fisher information and smooth it
    %Updated
    % FisherInformation = 256./(9.*Variance.^2);
    Variance=Variance.*Variance;
    FisherInformation =F_fi./Variance;

    % Compute embedding change probabilities and execute embedding
    FI = FisherInformation(:)';

    % Ternary embedding change probabilities
    beta = TernaryProbs(FI,Extimate_payload);
    beta(beta<=eps)=eps;

    COST=log(1./beta-2);
    COST=reshape(COST,size(X));

    % lf1=fspecial('average',[15 15]);
    if Spread_size>0
        lf1=fspecial('average',[Spread_size Spread_size]);

        COST=imfilter(COST,lf1,'symmetric','same');
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

    load('./MG_N/ixlnx3.mat');

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
