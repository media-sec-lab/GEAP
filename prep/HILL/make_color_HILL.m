function make_color_HILL(cover_dir,stego_dir,min_payload,max_payload)
% %By: Qinxinghong
% %Date: 2016.10.11
% %
    cdir=cover_dir;
    sdir=stego_dir;
    %cd(cdir);

    flist = dir([cdir filesep '*.ppm']);
    disp('Start make stego...');
    for k=min_payload:0.1:max_payload
        payload=single(k);
        t_path=sprintf('%s%sColorHILL_p%2d',sdir,filesep, uint8(k*100));
        if isdir(t_path)==0
            mkdir(t_path);
        end
        parfor i = 1:length(flist)
            imname = flist(i).name;
            t_file=[t_path filesep imname];
            if exist(t_file,'file')==0
                fprintf('[HILL-C]payload: %d, index: %d...\n',payload,i);
                X = imread([cdir filesep imname]);
                stego=make_hill(X,payload);
                imwrite(stego,t_file); 
            end
        end    
    end
end

function stego=make_hill(cover,payload)
   H=0; % It use simulator for embedding if H=0
   X=double(cover);
   stego=X;
   [s1,s2,s3]=size(cover);
    params.seed=0;%randi([j*100,(j+1)*100],1,1);
   for j=1:s3
        cost=f_cal_cost(X(:,:,j));% Obtain the costs for all pixels   
        %stego= f_embedding(x(:,:,j), cost, payload_l, H,params);% Embed the secret message  
        stego(:,:,j)= f_embedding(X(:,:,j), cost, payload, H,params);% Embed the secret message  
   end
   
   stego=uint8(stego);
end
