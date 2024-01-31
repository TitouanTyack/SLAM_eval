%% Prepare groundtruth pointcloud

ptCloud = pcread("/media/ce.debeunne/HDD/datasets/EUROC/" + ...
    "raw_data/V1_01_easy/mav0/pointcloud0/data.ply");

% Filter ceiling and points far away
points = ptCloud.Location - mean(ptCloud.Location);
dist = sqrt(sum(points.^2,2));
indices = logical((dist < 10) .* (points(:, 3) < 1.1));
points_inliers = points(indices,:);
ptCloudgt = pointCloud(points_inliers);
pcshow(ptCloudgt)

%% Prepare VIO pointcloud

% Read vio csv
vio = readmatrix("/media/ce.debeunne/HDD/datasets/EUROC/" + ...
    "cloud_v2/results.csv", 'OutputType', 'string');

% Extrinsic base / cam0
T_r_cam0 = [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
         0.0, 0.0, 0.0, 1.0];

% Read data
datapath = '/media/ce.debeunne/HDD/datasets/EUROC/cloud_v2/';
dir_scans = dir(datapath);
pts_vio = [];

for i=1:length(dir_scans)
    if (mod(i, 10) ~= 0)
        continue;
    end
    file = dir_scans(i);
    timestamp = erase(file.name, '.pcd');
    
    % Get associated pose
    idx = (vio(:,1) == timestamp);
    line = vio(idx, :);
    if (isempty(line))
        continue;
    end



    % extract pose from line (vio)
    t_w_r = [str2double(line(6)); str2double(line(10)); str2double(line(14))];
    R_w_r = [str2double(line(3)) str2double(line(4)) str2double(line(5));
        str2double(line(7)) str2double(line(8)) str2double(line(9));
        str2double(line(11)) str2double(line(12)) str2double(line(13))];
    T_w_r = [R_w_r, t_w_r; 0 0 0 1];
    T_w_cam0 = T_w_r * T_r_cam0;
    T_w_cam0_cheloue = [T_w_cam0(1:3, 1:3), zeros(3,1); T_w_cam0(1:3, 4)' 1];
    tform = affine3d(T_w_cam0');

    % Read point cloud
    vio_cloud = pcread(strcat(datapath,file.name));

    % Filter the cloud
    dist = sqrt(sum(vio_cloud.Location.^2,2));
    indices = (dist < 5);
    vio_cloud2 = pointCloud(vio_cloud.Location(indices, :));
    
    world_vio_cloud = pctransform(vio_cloud2, tform);
    pts_vio = [pts_vio; world_vio_cloud.Location];

end

% Display pc vio
ptcloud_vio_full = pointCloud(pts_vio);
pcshow(ptcloud_vio_full);

%% ICP cloud VIO -> GT

T_init_v2 = [-0.881925150736593	0.471206320285024	-0.0131389581659683	0;
-0.470975072996133	-0.881975981955647	-0.0173449666952413	0;
-0.0197613034622350	-0.00910884058591166	0.999763231924767	0;
0.454703301191330	1.31398046016693	-1.62553930282593	1];


T_init_v1 = [-0.979436191988437	-0.201696009764938	-0.00484411685361075	0;
0.201705927328260	-0.978394657349201	-0.0453719445382987	0;
0.00441188211950296	-0.0454160116637457	0.998958418143980	0;
0.749897301197052	1.39988183975220	-1.62852990627289	1];

T_init_v3 = [-0.990795485080859	0.133929273114349	-0.0196788349870422	0;
-0.133452007781089	-0.990768377367893	-0.0238450000416696	0;
-0.0226907109315948	-0.0209993383432257	0.999521965454770	0;
0.721335709095001	1.26596307754517	-1.72047352790833	1];

T_init_v21 = [0.999380811371693	0.0320404992230440	0.0145396104348015	0;
-0.0333032639860411	0.994720637337633	0.0970656801474610	0;
-0.0113528177089977	-0.0974897946666790	0.995171770834521	0;
-0.484087318181992	-0.161353424191475	-1.32720959186554	1];

T_init_v22 = [0.990527131116437	0.134829761527800	-0.0260180308362898	0;
-0.133955789188344	0.990443108642047	0.0328374037707960	0;
0.0301968386618480	-0.0290410734965965	0.999121998048784	0;
-0.483454793691635	-0.167129218578339	-1.45275104045868	1];

angle =  0;
R_z = [cos(angle) -sin(angle) 0 0;
    sin(angle) cos(angle) 0 0;
    0 0 1 0;
    0 0 0 1];

tform_init = rigid3d(T_init_v2 * R_z);
[tform, pctransformed, rmse] = pcregistericp(ptcloud_vio_full, ptCloudgt ,...
    'Verbose', true, 'InitialTransform', tform_init, ...
    'MaxIterations', 20, 'Tolerance', [0.01, 0.005]);

%% Display error accuracy

pointXYZ = zeros(length(pctransformed.Location), 3);
intensity = zeros(length(pctransformed.Location), 1);
dist_sum = 0;
n_dist = 0;
for i=1:length(pctransformed.Location)
    point = pctransformed.Location(i, :);
    [indices,dists] = findNearestNeighbors(ptCloudgt,point,1);
    if (dists(1) > 0.2)
        intensity(i) = 0.2;
    else
        intensity(i) = dists(1);
    end

    pointXYZ(i, :) = ptCloudgt.Location(indices(1), :);
    
end

% Build pc to display
pc_display = pointCloud(pointXYZ);
pc_display.Intensity = intensity;
pctransformed.Intensity = intensity;

pcshow(pctransformed, 'MarkerSize', 15, 'BackgroundColor', [1 1 1]);
colorbar();
colormap("jet");

%% Display error completeness

tform_init = rigid3d(T_init_v22 * R_z);
[tform, pctransformed, rmse] = pcregistericp(ptCloudgt, ptcloud_vio_full,...
    'Verbose', true, 'InitialTransform', tform_init.invert, ...
    'MaxIterations', 1, 'Tolerance', [0.01, 0.005]);

pointXYZ = zeros(length(pctransformed.Location), 3);
intensity = zeros(length(pctransformed.Location), 1);
dist_sum = 0;
n_dist = 0;
for i=1:length(pctransformed.Location)
    point = pctransformed.Location(i, :);
    [indices,dists] = findNearestNeighbors(ptcloud_vio_full,point,1);
    if (dists(1) > 0.2)
        intensity(i) = 0.2;
    else
        intensity(i) = dists(1);
    end

    if (dists(1) < 1)
        n_dist = n_dist + 1;
        dist_sum = dist_sum + dists(1);
        if (mod(i, 10000) == 0)
            disp(dist_sum / n_dist );
        end
    end
    pointXYZ(i, :) = ptCloudgt.Location(indices(1), :);
end