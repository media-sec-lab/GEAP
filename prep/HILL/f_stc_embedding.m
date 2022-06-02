function stego = f_stc_embedding(cover, rho, payload)
wetCost = 10^10;
%% Get embedding costs
h=10;
cover=int32(cover);
[r,t]=size(cover);

rhoP1 = rho;
rhoM1 = rho;
rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
rhoP1=reshape(rhoP1,1,r*t);
rhoM1=reshape(rhoM1,1,r*t);

m =floor(r*t*payload); 
msg = uint8(rand(1,m));        %Generating a random secert message

costs = zeros(3, r*t, 'single'); % Assign costs for +1 and -1
costs(1,:)=rhoM1;
costs(2,:)=0;
costs(3,:)=rhoP1;

cover1=reshape(cover,1,r*t);
[distortion, stegoA, n_msg_bits ] = stc_pm1_pls_embed(cover1, costs, msg, h);%embedding message
 extr_msg = stc_ml_extract(stegoA, n_msg_bits, h);
 if all(extr_msg~=msg)
     fprintf('Message was embedded and extract correctly\n');     
 end
stego = reshape(stegoA,r,t);

