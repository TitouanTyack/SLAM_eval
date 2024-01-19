%% Prepare groundtruth pointcloud

ptCloud = pcread("/media/ce.debeunne/HDD/datasets/EUROC/" + ...
    "V1_01_easy_raw/mav0/pointcloud0/data.ply");

% Filter ceiling and points far away
points = ptCloud.Location - mean(ptCloud.Location);
dist = sqrt(sum(points.^2,2));
indices = logical((dist < 10) .* (points(:, 3) < 1.1));
points_inliers = points(indices,:);
ptCloudgt = pointCloud(points_inliers);
pcshow(ptCloudgt)

%% Prepare VIO pointcloud

% Read gt csv
gt = readmatrix("/media/ce.debeunne/HDD/datasets/EUROC/" + ...
    "V1_01_easy_raw/mav0/state_groundtruth_estimate0/data.csv",...
    'OutputType', 'string');
vio = readmatrix("/media/ce.debeunne/HDD/datasets/EUROC/" + ...
    "cloud2/results.csv", 'OutputType', 'string');

% Extrinsic base / cam0
T_r_cam0 = [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
         0.0, 0.0, 0.0, 1.0];

% Read data
datapath = '/media/ce.debeunne/HDD/datasets/EUROC/cloud2/';
dir_scans = dir(datapath);
pts_vio = [];

for i=1:length(dir_scans)
    if (mod(i, 20) ~= 0)
        continue;
    end
    file = dir_scans(i);
    timestamp = erase(file.name, '.pcd');
    
    % Get associated pose
%     idx = (gt(:,1) == timestamp);
%     line = gt(idx, :);
%     if (length(line) == 0)
%         continue;
%     end
    idx = (vio(:,1) == timestamp);
    line = vio(idx, :);
    if (isempty(line))
        continue;
    end


    % extract pose from line (gt)
%     t_w_r = [str2double(line(2)); str2double(line(3)); str2double(line(4))];
%     q_w_r = [str2double(line(6)) str2double(line(7)) str2double(line(8)) str2double(line(5))];
%     R_w_r = quat2rotm(q_w_r);
%     T_w_r = [R_w_r, t_w_r; 0 0 0 1];
%     T_w_cam0 = T_w_r * T_r_cam0;
%     T_w_cam0_cheloue = [T_w_cam0(1:3, 1:3), zeros(3,1); T_w_cam0(1:3, 4)' 1];
%     tform = affine3d(T_w_cam0');

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
% dist = sqrt(sum(pts_vio.^2,2));
% indices = logical((dist < 10));
% ptcloud_vio_full = pointCloud(pts_vio(indices, :));
ptcloud_vio_full = pointCloud(pts_vio);
pcshow(ptcloud_vio_full);

%% ICP between the models

[tform, pctransformed, rmse] = pcregistericp(ptcloud_vio_full, ptCloudgt, 'Verbose', true);
