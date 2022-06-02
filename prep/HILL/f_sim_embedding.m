function stegoB = f_sim_embedding(cover, costmat, payload, params)

%% Get embedding costs
% inicialization
cover = double(cover);
seed = params.seed; %% seed for location selection
wetCost = 10^10;
% compute embedding costs \rho
rhoA = costmat;
rhoA(rhoA > wetCost) = wetCost; % threshold on the costs
rhoA(isnan(rhoA)) = wetCost; % if all xi{} are zero threshold the cost 
rhoP1 = rhoA;
rhoM1 = rhoA;
rhoP1(cover==255) = wetCost; % do not embed +1 if the pixel has max value
rhoM1(cover==0) = wetCost; % do not embed -1 if the pixel has min value
stegoB = f_EmbeddingSimulator_seed(cover, rhoP1, rhoM1, payload*numel(cover), seed); 

          
