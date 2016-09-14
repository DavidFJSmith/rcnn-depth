function imcls= run_all_segmantic(I, D, RD, C, out_file)
% function [E, ucm2, candidates, detection_scores_no_nms, cls] = run_all(I, D, RD, C, out_file)

% AUTORIGHTS

%% Compute the UCMs
p = get_paths();
edge_model_file = fullfile_ext(p.contours_model_dir, 'forest', 'modelNyuRgbd-3', 'mat');
model = load(edge_model_file);
model = model.model;
sc = [2 1 0.5];
[E, Es, O] = detectEdge(I, D, [], C, model, sc, [], []);
[ucm2 ucms] = contours_to_ucm(I, sc, Es, O);
if(~isempty(out_file)), save(out_file, 'E', 'Es', 'O', 'ucm2', 'ucms'); end

%compute the superpixel
sp = bwlabel(ucm2 < 0.20);
sp=sp(2:1:end, 2:1:end);
SuperPixelNum=max(max(sp));
I=imresize(I,2);
%divide the image according to the superpixel area.

%preload the model
net_file = fullfile_ext(p.snapshot_dir, sprintf('nyud2_finetune_color_iter_%d', 30000), 'caffemodel');
net_def_file = fullfile('nyud2_finetuning', 'imagenet_color_256_fc6.prototxt');
mean_file = fullfile_ext(p.mean_file_color, 'mat');
rcnn_model_RGB = rcnn_create_model(net_def_file, net_file, mean_file);
rcnn_model_RGB = rcnn_load_model(rcnn_model_RGB);

net_file = fullfile_ext(p.snapshot_dir, sprintf('nyud2_finetune_hha_iter_%d', 30000), 'caffemodel');
net_def_file = fullfile('nyud2_finetuning', 'imagenet_hha_256_fc6.prototxt');
mean_file = fullfile_ext(p.mean_file_hha, 'mat');
rcnn_model_HHA = rcnn_create_model(net_def_file, net_file, mean_file);
rcnn_model_HHA = rcnn_load_model(rcnn_model_HHA,0);%not using GPU

D=imresize(D,2);
RD=imresize(RD,2);
HHA = saveHHA([], C, [], D, RD);

% Load the detectors!
global RCNN_CONFIG_OVERRIDE;
conf_override.sub_dir = sprintf('rgb_hha_%d_%s', 30000, 'trainval');
RCNN_CONFIG_OVERRIDE = @() conf_override;
conf = rcnn_config();
dt = load([conf.cache_dir 'rcnn_model'], 'rcnn_model'); rcnn_model_detector = dt.rcnn_model; clear dt;

%最终输出类的结果
imcls=zeros(size(D,1),size(D,2));
%获取类颜色
load('/home/project/rcnn-depth-master/eccv14-data/benchmarkData/metadata/colordatabase.mat');

for i=1:SuperPixelNum
    row=[];
    col=[];
    [row,col]=find(sp==i);
    newim=zeros(size(I,1),size(I,2),size(I,3));
    %show the image
    for rr_=1:length(row)
        newim(row(rr_),col(rr_),:)=I(row(rr_),col(rr_),:);
    end
    %imshow(uint8(newim));
    
    boxTop=min(row);
    boxBottom=max(row);
    boxLeft=min(col);
    boxRight=max(col);
    %imagesc(abc);
    %rectangle('Position',[boxLeft,boxTop,boxRight-boxLeft,boxBottom-boxTop]);
    
    %cut the image to the size of the sp
    newim=newim(boxTop:boxBottom,boxLeft:boxRight,:);
    %imshow(uint8(newim));
    %pause;close all;
    
    boxes=[1,1,size(newim,2),size(newim,1)];
    %box is the size of the image
    %boxes = candidates.bboxes(1:2000, [2 1 4 3]);
    
    feat={};
    % Compute the RGB Features
    feat{1} = rcnn_features(newim, boxes, rcnn_model_RGB);
    
    % Compute the HHA Features
    feat{2} = rcnn_features(HHA, boxes, rcnn_model_HHA);
    feat = cat(2, feat{:});
    
    %detectors
    feat = rcnn_scale_features(feat, rcnn_model_detector.training_opts.feat_norm_mean);
    detection_scores_no_nms = bsxfun(@plus, feat*rcnn_model_detector.detectors.W, rcnn_model_detector.detectors.B);
    cls = rcnn_model_detector.classes;
    %extract the max score class.
    [max_score,index_]=max(detection_scores_no_nms);
    predictclass=cls(index_);
    
    %class2color
    classnum=databasemap.cls2num_map(predictclass);
    %对超像素的每个点进行赋值。
    for rr_=1:length(row)
        imcls(row(rr_),col(rr_))=classnum;
    end
    %show the image
    
end

end
