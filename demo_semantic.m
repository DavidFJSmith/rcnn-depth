startup();

% Load the detectors!
global RCNN_CONFIG_OVERRIDE;
conf_override.sub_dir = sprintf('rgb_hha_%d_%s', 30000, 'trainval');
RCNN_CONFIG_OVERRIDE = @() conf_override;
conf = rcnn_config();
dt = load([conf.cache_dir 'rcnn_model'], 'rcnn_model'); rcnn_model_detector = dt.rcnn_model; clear dt;
%所要判断的类的种类。
cls = rcnn_model_detector.classes;

%获取类颜色
load('/home/project/rcnn-depth-master/eccv14-data/benchmarkData/metadata/colordatabase.mat');

%获取全部的图片：
%ImagePath=[''];
datadir='/home/project/rcnn-depth-master/eccv14-data/';
imdir=[datadir 'data/images'];
depthdir=[datadir 'data/depth'];
rawdepth=[datadir 'data/rawdepth'];
gt_dir=[datadir 'benchmarkData/groundTruth'];

C = cropCamera(getCameraParam('color'));
out_file = fullfile('demo-data', 'output.mat');

%获取对应的颜色的GT。
allname=dir(imdir);
for ii=3:length(allname)
    imageName=allname(ii).name;
    I = imread(fullfile(imdir,imageName));
    D = imread(fullfile(depthdir,imageName));
    RD = imread(fullfile(rawdepth,imageName));
    GTname=strrep(imageName,'.png','.mat');
    GT= load(fullfile(gt_dir,GTname));
    GT=GT.groundTruth;
    
    backgroundcolor=[0,0,0];
    %画出整个的GT：
    gt_seg=GT{1}.SegmentationClass;
    Gt_color_image=zeros(size(I,1),size(I,2),size(I,3));
    
    %把GT映射到color：
    for gt_ii=1:size(gt_seg,1)
        for gt_kk=1:size(gt_seg,2)
            GtClass=gt_seg(gt_ii,gt_kk);
            t_f=databasemap.num2cls_map.isKey(num2str(GtClass));
            if t_f==0
                %这就是背景。
                Gt_color_image(gt_ii,gt_kk,:)=backgroundcolor;
            else
                %取得相应的颜色。
                cls__=databasemap.num2cls_map(num2str(GtClass));
                cls_color=databasemap.name2color_map(cls__);
                Gt_color_image(gt_ii,gt_kk,:)=cls_color;
            end
        end
    end
    imshow(I);
    imshow(uint8(Gt_color_image));
end

classnum=databasemap.cls2num_map(cls_);
%读取数据集，并且对一张图片进行处理，处理出GT
cls_img=run_all_segmantic(I, D, RD, C, out_file);
