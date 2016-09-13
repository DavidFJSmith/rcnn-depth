function [E, ucm2, candidates, detection_scores_no_nms, cls] = run_all_segmantic(I, D, RD, C, out_file)
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

%% Compute the regions
%   params = nyud_params('root_cache_dir', p.cache_dir, 'feature_id', 'depth', 'depth_features', true, 'camera_matrix', C);  
%   rf = loadvar(params.files.trained_classifier,'rf');
%   n_cands = loadvar(params.files.pareto_point,'n_cands');
%    
%   mcg_cache_obj = cache_mcg_features(params, {ucm2, ucms(:,:,1), ucms(:,:,2), ucms(:,:,3)}, [], []);
%   candidates = compute_mcg_cands(params, rf, n_cands, mcg_cache_obj, D, RD);
%   if(~isempty(out_file)), save(out_file, '-append', 'candidates'); end


% Display the superpixels and the regions
%   figure(1); 
%   subplot(2,3,1); imagesc(Es{2}); axis image; title('Edge Signal');
%   subplot(2,3,2); imagesc(ucm2(3:2:end, 3:2:end)); axis image; title('Multi UCM');
%   sp = bwlabel(ucm2 < 0.20); sp = sp(2:2:end, 2:2:end);
%   for i = 1:3, csp(:,i) = accumarray(sp(:), linIt(I(:,:,i)), [], @mean); end %caculate the mean color that in one sp.
%   subplot(2,3,3); imagesc(ind2rgb(sp, im2double(uint8(csp)))); axis image; title('Superpixels');

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

  %if(~isempty(out_file)), save(out_file, '-append', 'detection_scores_no_nms', 'cls'); end

end  
% Visualize some detections
%   cls_id = [2 5 16 17];
%   cols = lines(length(cls_id));
%   Idet = I;
%   for i = 1:length(cls_id),
%     dt = load(fullfile(conf.cache_dir, 'pr-curves', sprintf('%s_pr_nyud2_test_release.mat', cls{cls_id(i)})));
%     bbox = cat(2, boxes, detection_scores_no_nms(:,cls_id(i)));
%     keep = false(size(bbox(:,1)));
%     keep(rcnn_nms(bbox, 0.3)) = 1;
%     thresh = dt.thresh(find(dt.prec > 0.8, 1, 'last'));
%     ind = bbox(:,5) > thresh;
%     keep = find(keep & ind);
%     bbox = bbox(keep,:);
%     if(size(bbox,1) > 0)
%       Idet = draw_rect_vec(Idet, bbox(:,1:4)', im2uint8(cols(i,:)), 2);
%     end
%   end
%   figure(1); subplot(2,3,4); imagesc(Idet); axis image; title(['detections - ', sprintf('%s, ', cls{cls_id})]);
%   figure(1); subplot(2,3,5); plot([1:length(cls_id)], 1); legend(cls(cls_id)); axis image;
  
  %% Do instance segmentation
  % [] = instance_segmentation(I, D, detections, sp);
  %% Visualize the instance segmentations
end
