startup();




I = imread(fullfile('demo-data', 'images.png'));
D = imread(fullfile('demo-data', 'depth.png'));
RD = imread(fullfile('demo-data', 'rawdepth.png'));

%读取数据集，并且对一张图片进行处理，处理出GT




C = cropCamera(getCameraParam('color'));
out_file = fullfile('demo-data', 'output.mat');



cls_img=run_all_segmantic(I, D, RD, C, out_file);



